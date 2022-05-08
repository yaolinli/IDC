'''
    Revise eval for evaluation
'''

from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from ciderD.ciderD import CiderD
# from spice.spice import Spice

class Evaluator:
    def __init__(self, references, candidates):
        self.references = references
        self.candidates = candidates
        self.eval = {}
        self.imgToEval = {}

    def evaluate(self):
        print('Setting up scores...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (CiderD(), "CIDErD")
            # (Spice(), "SPICE")
        ]

        for scorer, method in scorers:
            print('\nCompute ', scorer.method())
            score, scores = scorer.compute_score(self.references, self.candidates)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, self.references.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, self.references.keys(), method)
                print("%s: %0.3f"%(method, score))
            self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
