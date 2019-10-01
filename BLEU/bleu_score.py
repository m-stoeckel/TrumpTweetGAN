from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# reference can be multiple, while hypothesis is only one

def calc_bleu(refer_file, hypo_file):
    dev_ret = []
    with open(hypo_file, "r") as f:
        for i in f:
            i = i.strip()
            idxs = i.split()
            dev_ret.append(idxs)
    dev_ref = []

    with open(refer_file, "r") as f:
        for i in f:
            i = i.strip()
            idxs = i.split()
            dev_ref.append(idxs)
    bleu = 0
    smooth = SmoothingFunction()
    for ref, hyp in zip(dev_ref, dev_ret):
        ref_remove = [x for x in ref if x != 0]
        hyp_remove = [x for x in hyp if x != 0]
        s = sentence_bleu([ref_remove], hyp_remove, smoothing_function=smooth.method1)
        bleu += s
        # print(s)
    # print(i)
    # print(len(dev_ret))
    return bleu / len(dev_ret)


if __name__ == '__main__':

    ref = "../dataset/trump_tweets_2016-2019.txt"
    #results = "../_results/leakgan_samples_ADV.txt"
    results = "../_results/relgan_samples_ADV.txt"

    print(calc_bleu(ref, results))
