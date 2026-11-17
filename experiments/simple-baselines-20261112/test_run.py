from run import predictions, basic_score

def main():
    ids = [str(i) for i in range(10)]
    histories = [['1','1','2','3','2','4']]
    assert predictions('recent_history',ids,histories,ids,0) == [['4','2','3','1','0']]
    for seed in range(10):
        row = predictions('uniform_random',ids,histories,ids,seed)
        assert len(set(row[0])) == 5
        assert row == predictions('uniform_random',ids,histories,ids,seed)
    pool = [['3','4','5','6','7']]
    assert set(predictions('uniform_random',ids,histories,ids,1,pool)[0]) == set(pool[0])
    itemc = {s:list(range(8)) for s in ids}
    scores,_ = basic_score([['0','1','2','3','4']],[['0','0','1','8','9']],histories,itemc,ids)
    assert scores['recall_at_5'] == .4 and scores['cue_f1'] == 1 and scores['diversity'] == .5
    print('PASS deterministic random, eligible pool, recent fallback and repeated-target scoring')

if __name__ == '__main__':main()
