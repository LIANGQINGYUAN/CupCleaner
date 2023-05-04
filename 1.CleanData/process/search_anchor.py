import numpy as np

# Most good data is on the right, expand the leftmost area.
def get_delta_changes_right_expand(source, delta=0.1, lambda_num=2):
    avg = np.mean(source)
    std = np.std(source)
    length = len(source)
    new_lambda_num = lambda_num-delta
    tail = [i for i in list(source) if i<avg-new_lambda_num*std]
    prev_tail = [i for i in list(source) if i<avg-(new_lambda_num+0.1)*std]
    # print("Delta: ", delta)
    # print("New anchor: ", avg-new_lambda_num*std, " Area of this anchor: ", len(tail)/length)
    
    if delta==0:
        ratio_delta = len(tail)/length # no changes
        return ratio_delta, ratio_delta , 0
    else:
        ratio_delta = len(tail)/length
        ratio_prev = len(prev_tail)/length # previous
        return ratio_delta, ratio_prev, ratio_delta-ratio_prev

# Most good data is on the right, shrink the leftmost area.
def get_delta_changes_right_shrink(source, delta=0.1, lambda_num=2):
    avg = np.mean(source)
    std = np.std(source)
    length = len(source)
    new_lambda_num = lambda_num+delta
    tail = [i for i in list(source) if i<avg-new_lambda_num*std]
    prev_tail = [i for i in list(source) if i<avg-(new_lambda_num-0.1)*std]
    # print("Delta: ", delta)
    # print("New anchor: ", avg-new_lambda_num*std, " Area of this anchor: ", len(tail)/length)
    
    if delta==0:
        ratio_delta = len(tail)/length # no changes
        return ratio_delta, ratio_delta , 0
    else:
        ratio_delta = len(tail)/length
        ratio_prev = len(prev_tail)/length # previous
        return ratio_delta, ratio_prev, ratio_prev-ratio_delta

def heuristic_search(source, threshold):
    lambda_num = 2
    # anchor, delta = get_anchor(source, lambda_num=lambda_num)
    # return anchor
    # for i in range(0,100):
    #     lambda_num = lambda_num+0.01*i
    #     anchor, delta = get_anchor(source, threshold, lambda_num=lambda_num)
    #     if anchor == delta == 0:
    #         return None
    #     if delta-0.01==0.0:
    #         continue
    #     else:
    #         return anchor
    anchor, delta = get_anchor(source, threshold, lambda_num=lambda_num)
    return anchor

def get_anchor(source, threshold, lambda_num=2):
    avg = np.mean(source)
    std = np.std(source)
    var = np.var(source)
    print("Lambda num:", lambda_num)
    print("Avg, std and var: ", avg, std, var)
    ratio, _, _ = get_delta_changes_right_expand(source, delta=0, lambda_num=lambda_num)
    # print("Anchor: ", avg-lambda_num*std)
    # print(f"Area of avg-{lambda_num}*std: ", ratio)
    for i in range(1,201):
        delta = i*0.01
        ratio_delta, ratio_prev, ratio_change = get_delta_changes_right_expand(source, delta=delta, lambda_num=lambda_num)
        # print("Ratio of current delta: ", ratio_delta, "Ration of previous delta: ", ratio_prev, "Ration change: ", ratio_change)
        # print("*"*30)
        # if ratio_change>max(min(0.05*std,0.003), 0.002):
        if ratio_change>threshold:
            # print('Selected delta: ',delta-0.01)
            new_lambda_num = lambda_num-(delta-0.01)
            tail = [i for i in list(source) if i<avg-new_lambda_num*std]
            print("New anchor: ", avg-new_lambda_num*std, " Area of this anchor: ", len(tail)/len(source))
            print(get_delta_changes_right_expand(source, delta=delta-0.01, lambda_num=lambda_num))
            return avg-new_lambda_num*std, delta
    return 0,0