# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric 
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

# ReinforceCIDEr is an alternative implementation of CIDEr where the corpus is initialized in the constructor
# so it doesn't need to be processed again every time we need to compute the cider score
# in the Self Critical Learning Process. --- Jia Cheng

from eval.cider.reinforce_cider_scorer import ReinforceCiderScorer
import pdb


class ReinforceCider:

    # The batch_ref_sentences will be a small sample of the original corpus, note however that there's no need of
    # correspondence of img_ids between img_ids in the corpus and the ones in the batch_ref_sentences, the img_ids
    # consistency is required between batch_ref_sentences and batch_test_sentences only.
    def __init__(self,  corpus, n=4, sigma=6.0):
        '''
        Corpus represents the collection of reference sentences for each image, this must be a dictionary with image
        ids as keys and a list of sentences as value.

        :param corpus: a dictionary with
        :param n: number of n-grams
        :param sigma: length penalty coefficient
        '''
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self.cider_scorer = ReinforceCiderScorer(corpus, n=self._n, sigma=self._sigma)

    def compute_score(self, hypo, refs):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus 
        """

        # assert(hypo.keys() == refs.keys())

        (score, scores) = self.cider_scorer.compute_score(refs, hypo)

        return score, scores

    def method(self):
        return "Reinforce CIDEr"