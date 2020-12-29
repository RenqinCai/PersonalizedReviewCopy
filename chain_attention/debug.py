import random
import numpy as np
import copy
import datetime
 
# Driver code

a = [i for i in range(10)]

b = [0 for i in range(10)]

a = np.array(a)
b = np.array(b==1)
c = a[b]

print("c", c)

exit()
start_time = datetime.datetime.now()
for word in range(100):
    n = 8
    result = [[0], [1]]

    for i in range(1, n):
        next_result = []
        for tmp in result:
            tmp_0 = copy.deepcopy(tmp)
            tmp_0.append(0)
            next_result.append(tmp_0)

            tmp_1 = copy.deepcopy(tmp)
            tmp_1.append(1)
            next_result.append(tmp_1)

        result = next_result
end_time = datetime.datetime.now()
duration = end_time-start_time
print("duration", duration)


print(result[-1])
# for i in range(10):
#     print(result[i])

exit()

random_freq = 10
good_freq = 0

for random_i in range(random_freq):
    n = 9

    a = np.array([i for i in range(n)])

    # result = [[0], [1]]

    # start_time = datetime.datetime.now()
    # for i in range(1, n): 
        
    #     next_result = []
    #     for tmp in result:
    #         tmp_0 = copy.deepcopy(tmp)
    #         tmp_0.append(0)
    #         next_result.append(tmp_0)

    #         tmp_1 = copy.deepcopy(tmp)
    #         tmp_1.append(1)
    #         next_result.append(tmp_1)

    #     result = next_result
    # end_time = datetime.datetime.now()
    # duration = end_time-start_time
    # print("duration", duration)

    start_time = datetime.datetime.now()
    result = []
    threshold = 100
    for i in range(threshold):
        b = np.array([random.randint(0, 1)==1 for j in range(n)])
        c = a[b]
        result.append(c)
    end_time = datetime.datetime.now()
    duration = end_time-start_time
    # print("duration", duration)

    tmp_set = set()
    
    for i in result:
        i = [str(j) for j in i]
        new_i = "".join(i)
        # print(new_i)
        tmp_set.add(new_i)

    # print(result)
    if len(tmp_set) == len(result):
        good_freq += 1
    else:
        print(len(tmp_set))
        print(len(result))

print("good_freq", good_freq)
    

# [A1, A2, A3]

# []->[A1, A2, A3]
# [A1]->[A2, A3]
# [A1, A2]->[A3]
# [A2]->[A1, A3]
# [A3]->[A1, A2]
# [A1, A3]->[A2]
# [A2, A3]->[A1]
