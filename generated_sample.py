import codecs, data, json, nltk, os, re, rouge, shutil, time
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from statistics import mean
from tensorboardX import SummaryWriter
from tqdm import tqdm
FLAGS = tf.app.flags.FLAGS

class Bleu(object):
    bleu1_weight = (1, 0, 0, 0)
    bleu2_weight = (0.5, 0.5, 0, 0)
    bleu3_weight = (0.33, 0.33, 0.33, 0)
    bleu4_weight = (0.25, 0.25, 0.25, 0.25)

    @staticmethod
    def summary_dict(bleu_scores):
        """
        Generate a dict for add_scalars() in tensorboardX.
        # Args:
        - bleu_scores: a tuple (list) with bleu-n scores
        """
        summary_dict = {}
        for i in range(len(bleu_scores)):
            summary_dict["bleu%s" % str(i + 1)] = bleu_scores[i]
        return summary_dict

class Rouge(object):
    @staticmethod
    def summary_dict(rouge_scores):
        """
        Generate a dict for add_scalars() in tensorboardX.
        # Args:
        - rouge_scores: a dict, e.g. {"rouge-1": {"f": _, "p": _, "r": _}, "rouge-2" : { ..     }, "rouge-l": { ... }}
        """
        f_summary_dict = {}
        p_summary_dict = {}
        r_summary_dict = {}
        for key in rouge_scores:
            f_summary_dict[key] = rouge_scores[key]["f"]
            p_summary_dict[key] = rouge_scores[key]["p"]
            r_summary_dict[key] = rouge_scores[key]["r"]
        return f_summary_dict, p_summary_dict, r_summary_dict

class Generated_sample(object):
    def __init__(self, model, vocab, batcher, sess):
        self._model = model
        self._vocab = vocab
        self._sess = sess
        self.batches = batcher.get_batches(mode='train')
        self.test_batches = batcher.get_batches(mode='test')
        self.current_batch = 0
        if not os.path.exists("discriminator_train"): os.mkdir("discriminator_train")
        if not os.path.exists("discriminator_test"): os.mkdir("discriminator_test")
        self.train_sample_whole_positive_dir = os.path.join("discriminator_train","positive")
        self.train_sample_whole_negative_dir = os.path.join("discriminator_train","negative")
        self.test_sample_whole_positive_dir = os.path.join("discriminator_test", "positive")
        self.test_sample_whole_negative_dir = os.path.join("discriminator_test", "negative")
        if not os.path.exists(self.train_sample_whole_positive_dir): os.mkdir(self.train_sample_whole_positive_dir)
        if not os.path.exists(self.train_sample_whole_negative_dir): os.mkdir(self.train_sample_whole_negative_dir)
        if not os.path.exists(self.test_sample_whole_positive_dir): os.mkdir(self.test_sample_whole_positive_dir)
        if not os.path.exists(self.test_sample_whole_negative_dir): os.mkdir(self.test_sample_whole_negative_dir)

    def check_dir(self, positive_dir, negative_dir):
        if not os.path.exists(positive_dir): os.mkdir(positive_dir)
        if not os.path.exists(negative_dir): os.mkdir(negative_dir)
        shutil.rmtree(negative_dir)
        shutil.rmtree(positive_dir)
        if not os.path.exists(positive_dir): os.mkdir(positive_dir)
        if not os.path.exists(negative_dir): os.mkdir(negative_dir)

    def write_negtive_to_json(self, positive, negative, counter, positive_dir, negtive_dir):
        positive_file = os.path.join(positive_dir, "%06d.txt" % (counter // 1000))
        negative_file = os.path.join(negtive_dir, "%06d.txt" % (counter // 1000))
        write_positive_file = codecs.open(positive_file, "a", "utf-8")
        write_negative_file = codecs.open(negative_file, "a", "utf-8")
        dict = {"example": str(positive),
                "label": str(1)
                }
        string_ = json.dumps(dict)
        write_positive_file.write(string_ + "\n")

        dict = {"example": str(negative),
                "label": str(0)
                }
        string_ = json.dumps(dict)
        write_negative_file.write(string_ + "\n")
        write_negative_file.close()
        write_positive_file.close()

    def process_generated_summary(self, batch, neg_summary, positive_dir, negative_dir, counter, doc2doc_bleu = False, sen2sen_bleu = False, group_sen_bleu = False, rouge = False):
        doc_bleu_hyp, doc_bleu_ref, sen_bleu_hyp, sen_bleu_ref = [], [], [], []
        group_bleu1, group_bleu2, group_bleu3, group_bleu4 = 0, 0, 0, 0
        rouge_hyp, rouge_ref = [], []

        for i in range(FLAGS.batch_size):
            decoded_words_all = []
            pos_summary = batch.original_review_output[i]

            for j in range(FLAGS.max_dec_sen_num):
                output_ids = [int(t) for t in neg_summary['generated'][i][j]][1:]
                decoded_words = data.outputids2words(output_ids, self._vocab, None)

                # In the sentence, if there is a [STOP_DECODING] token, remove the words after the token.
                try:
                    fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP_DECODING] token
                    decoded_words = decoded_words[:fst_stop_idx]
                except ValueError:
                    decoded_words = decoded_words

                if len(decoded_words) < 2:
                    continue

                decoded_output = ' '.join(decoded_words).strip() # single string
                decoded_words_all.append(decoded_output)
            
            decoded_words_all = ' '.join(decoded_words_all).strip()

            # In the sample, if there is a [STOP_DECODING_DOCUMENT] token, remove the words after the token.
            try:
                fst_stop_idx = decoded_words_all.index(data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP_DECODING_DOCUMENT] symbol
                decoded_words_all = decoded_words_all[:fst_stop_idx]
            except ValueError:
                decoded_words_all = decoded_words_all
            
            decoded_words_all = decoded_words_all.replace("[UNK] ", "")
            decoded_words_all = decoded_words_all.replace("[UNK]", "")
            decoded_words_all, _ = re.subn(r"(! ){2,}", "! ", decoded_words_all)
            decoded_words_all, _ = re.subn(r"(\. ){2,}", ". ", decoded_words_all)

            # Write to file
            self.write_negtive_to_json(pos_summary, decoded_words_all, counter, positive_dir, negative_dir)
            counter += 1

            # Calculate bleu scores
            if doc2doc_bleu:
                doc_bleu_ref.append([pos_summary.split()])
                doc_bleu_hyp.append(decoded_words_all.split())
            
            if sen2sen_bleu:
                pos_sentences = nltk.sent_tokenize(pos_summary)
                pos_sentences = [sen.split() for sen in pos_sentences]
                neg_sentences = nltk.sent_tokenize(decoded_words_all)
                neg_sentences = [sen.split() for sen in neg_sentences]
                neg_sen_num = len(neg_sentences)
                
                sen_bleu_ref.extend([pos_sentences] * neg_sen_num)
                sen_bleu_hyp.extend(neg_sentences)

                if group_sen_bleu:
                    for neg_sen in neg_sentences:
                        group_bleu1 += sentence_bleu(pos_sentences, neg_sen, weights=Bleu.bleu1_weight) / (neg_sen_num * FLAGS.batch_size)
                        group_bleu2 += sentence_bleu(pos_sentences, neg_sen, weights=Bleu.bleu2_weight) / (neg_sen_num * FLAGS.batch_size)
                        group_bleu3 += sentence_bleu(pos_sentences, neg_sen, weights=Bleu.bleu3_weight) / (neg_sen_num * FLAGS.batch_size)
                        group_bleu4 += sentence_bleu(pos_sentences, neg_sen) / (neg_sen_num * FLAGS.batch_size)
            
            # Calculate rouge scores
            if rouge:
                rouge_ref.append(pos_summary)
                rouge_hyp.append(decoded_words_all)
                        
        return counter, (doc_bleu_hyp, doc_bleu_ref), (sen_bleu_hyp, sen_bleu_ref), (group_bleu1, group_bleu2, group_bleu3, group_bleu4), (rouge_hyp, rouge_ref)

    def generator_train_sample_example(self, positive_dir, negative_dir, num_batch):
        self.check_dir(positive_dir, negative_dir)
        counter = 0
        for _ in range(num_batch):
            batch = self.batches[self.current_batch]
            decode_result = self._model.run_eval_given_step(self._sess, batch)          
            counter, _, _, _, _ = self.process_generated_summary(batch, decode_result, positive_dir, negative_dir, counter)
            
            self.current_batch += 1
            if self.current_batch >= len(self.batches):
                self.current_batch = 0

    def generator_train_max_example(self, positive_dir, negative_dir, num_batch):
        self.check_dir(positive_dir, negative_dir)
        counter = 0
        for _ in range(num_batch):
            batch = self.batches[self.current_batch]
            decode_result = self._model.max_generator(self._sess, batch)
            counter, _, _, _, _ = self.process_generated_summary(batch, decode_result, positive_dir, negative_dir, counter)

            self.current_batch +=1
            if self.current_batch >= len(self.batches):
                self.current_batch = 0

    def generator_test_example(self, run_sess_func, positive_dir, negative_dir):
        self.check_dir(positive_dir, negative_dir)
        
        counter = 0
        doc_bleu_hyp_list, doc_bleu_ref_list, sen_bleu_hyp_list, sen_bleu_ref_list = [], [], [], []
        group_bleu1_list, group_bleu2_list, group_bleu3_list, group_bleu4_list = [], [], [], []
        rouge_hyp_list, rouge_ref_list = [], []
        
        for batch in tqdm(self.test_batches, ascii=True):
            decode_result = run_sess_func(self._sess, batch)
            counter, (doc_bleu_hyp, doc_bleu_ref), (sen_bleu_hyp, sen_bleu_ref), (group_bleu1, group_bleu2, group_bleu3, group_bleu4), (rouge_hyp, rouge_ref) = self.process_generated_summary(batch, decode_result, positive_dir, negative_dir, counter, doc2doc_bleu=True, sen2sen_bleu=True, group_sen_bleu=True, rouge=True)
            
            doc_bleu_hyp_list.extend(doc_bleu_hyp)
            doc_bleu_ref_list.extend(doc_bleu_ref)
            sen_bleu_hyp_list.extend(sen_bleu_hyp)
            sen_bleu_ref_list.extend(sen_bleu_ref)

            group_bleu1_list.append(group_bleu1)
            group_bleu2_list.append(group_bleu2)
            group_bleu3_list.append(group_bleu3)
            group_bleu4_list.append(group_bleu4)

            rouge_hyp_list.extend(rouge_hyp)
            rouge_ref_list.extend(rouge_ref)
        
        # Calculate bleu score
        doc_bleu1 = corpus_bleu(doc_bleu_ref_list, doc_bleu_hyp_list, weights=Bleu.bleu1_weight)
        doc_bleu2 = corpus_bleu(doc_bleu_ref_list, doc_bleu_hyp_list, weights=Bleu.bleu2_weight)
        doc_bleu3 = corpus_bleu(doc_bleu_ref_list, doc_bleu_hyp_list, weights=Bleu.bleu3_weight)
        doc_bleu4 = corpus_bleu(doc_bleu_ref_list, doc_bleu_hyp_list)

        sen_bleu1 = corpus_bleu(sen_bleu_ref_list, sen_bleu_hyp_list, weights=Bleu.bleu1_weight)
        sen_bleu2 = corpus_bleu(sen_bleu_ref_list, sen_bleu_hyp_list, weights=Bleu.bleu2_weight)
        sen_bleu3 = corpus_bleu(sen_bleu_ref_list, sen_bleu_hyp_list, weights=Bleu.bleu3_weight)
        sen_bleu4 = corpus_bleu(sen_bleu_ref_list, sen_bleu_hyp_list)
        
        group_bleu1 = mean(group_bleu1_list)
        group_bleu2 = mean(group_bleu2_list)
        group_bleu3 = mean(group_bleu3_list)
        group_bleu4 = mean(group_bleu4_list)

        # Calculate rouge score
        rouge_model = rouge.Rouge()
        rouge_scores = rouge_model.get_scores(rouge_hyp_list, rouge_ref_list, avg=True)   
        return (doc_bleu1, doc_bleu2, doc_bleu3, doc_bleu4), (sen_bleu1, sen_bleu2, sen_bleu3, sen_bleu4), (group_bleu1, group_bleu2, group_bleu3, group_bleu4), rouge_scores

    def generator_test_sample_example(self, positive_dir, negative_dir, num_batch):
        doc_bleu_scores, sen_bleu_scores, group_bleu_scores, rouge_scores = self.generator_test_example(self._model.run_eval_given_step, positive_dir, negative_dir)
        
        with SummaryWriter(FLAGS.log_root) as summary_writer:
            summary_writer.add_scalars("Adversarial/Bleu/Sample/Doc2Doc", Bleu.summary_dict(doc_bleu_scores))
            summary_writer.add_scalars("Adversarial/Bleu/Sample/Sen2Sen", Bleu.summary_dict(sen_bleu_scores))
            summary_writer.add_scalars("Adversarial/Bleu/Sample/Group_Sen", Bleu.summary_dict(group_bleu_scores))

            f_summary_dict, p_summary_dict, r_summary_dict = Rouge.summary_dict(rouge_scores)
            summary_writer.add_scalars("Adversarial/Rouge/Sample/F1", f_summary_dict)
            summary_writer.add_scalars("Adversarial/Rouge/Sample/Precision", p_summary_dict)
            summary_writer.add_scalars("Adversarial/Rouge/Sample/Recall", r_summary_dict)

    def generator_test_max_example(self, positive_dir, negative_dir, num_batch):
        doc_bleu_scores, sen_bleu_scores, group_bleu_scores, rouge_scores = self.generator_test_example(self._model.max_generator, positive_dir, negative_dir)
        
        with SummaryWriter(FLAGS.log_root) as summary_writer:
            summary_writer.add_scalars("Adversarial/Bleu/Max/Doc2Doc", Bleu.summary_dict(doc_bleu_scores))
            summary_writer.add_scalars("Adversarial/Bleu/Max/Sen2Sen", Bleu.summary_dict(sen_bleu_scores))
            summary_writer.add_scalars("Adversarial/Bleu/Max/Group_Sen", Bleu.summary_dict(group_bleu_scores))

            f_summary_dict, p_summary_dict, r_summary_dict = Rouge.summary_dict(rouge_scores)
            summary_writer.add_scalars("Adversarial/Rouge/Max/F1", f_summary_dict)
            summary_writer.add_scalars("Adversarial/Rouge/Max/Precision", p_summary_dict)
            summary_writer.add_scalars("Adversarial/Rouge/Max/Recall", r_summary_dict)

    def generator_pretrain_train_example(self):
        counter = 0
        for batch in tqdm(self.batches, ascii=True):
            decode_result = self._model.run_eval_given_step(self._sess, batch)
            counter, _, _, _, _ = self.process_generated_summary(batch, decode_result, self.train_sample_whole_positive_dir, self.train_sample_whole_negative_dir, counter)

    def generator_pretrain_test_example(self):
        counter = 0
        for batch in tqdm(self.test_batches, ascii=True):
            decode_result = self._model.run_eval_given_step(self._sess, batch)
            counter, _, _, _, _ = self.process_generated_summary(batch, decode_result, self.test_sample_whole_positive_dir, self.test_sample_whole_negative_dir, counter)

    def compute_BLEU(self, train_step):

        counter = 0
        step = 0


        t0 = time.time()
        batches = self.test_batches
        list_hop = []
        list_ref = []

        #tf.logging.info(len(batches))

        while step <  100:
            #tf.logging.info(step)


            batch = batches[step]
            step += 1

            decode_result = self._model.run_eval_given_step(self._sess, batch)

            #tf.logging.info(step)

            for i in range(FLAGS.batch_size):

                #tf.logging.info("i: " + str(i))

                decoded_words_all = []
                original_review = batch.original_review_output[i]  # string

                for j in range(FLAGS.max_dec_sen_num):

                    #tf.logging.info("j: " + str(j))

                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if len(decoded_words)<2:
                        continue

                    '''if j>0:
                        new_set1 =set(decoded_words_all[j-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set1):
                            continue'''
                    if decoded_words[-1] !='.' and decoded_words[-1] !='!' and decoded_words[-1] !='?':
                        decoded_words.append('.')

                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)
                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
                decoded_words_all,_ = re.subn(r"(\. ){2,}", "", decoded_words_all)


                list_hop.append(decoded_words_all)
                list_ref.append(original_review)
                #self.write_negtive_to_json(original_review, decoded_output, counter)

                #counter += 1  # this is how many examples we've decoded
        '''file_temp = open(train_step+"_temp_result.txt",'w')
        for hop in list_hop:
            file_temp.write(hop+"\n")
        file_temp.close()'''
        '''new_ref_list =[]
        for ref in list_ref:
            sens = nltk.sent_tokenize(ref)
            for sen in sens:
                new_ref_list.append(nltk.word_tokenize(sen))
        t0 = time.time()
        new_sen_list =[]
        new_ref_ref =[]
        for hop in list_hop:
            sens = nltk.sent_tokenize(hop)
            for sen in sens:
                new_sen_list.append(nltk.word_tokenize(sen))

                new_ref_ref.append(new_ref_list)'''

        #print (new_sen_list)



        #bleu_score = corpus_bleu(new_ref_ref, new_sen_list)
        t1 = time.time()
        tf.logging.info('seconds for test generator: %.3f ', (t1 - t0))
        return 0

