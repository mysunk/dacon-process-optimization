'# -*- coding: utf-8 -*-'
"""
Created on Fri Jun  5 22:54:56 2020

@author: guseh
"""

import numpy as np
import multiprocessing
import warnings
from copy import deepcopy
from module.genome import Genome, genome_score
warnings.filterwarnings(action='ignore')
np.random.seed(76)

#%% 변수 선택
CPU_CORE = multiprocessing.cpu_count()          # 멀티프로세싱 CPU 사용 수
N_POPULATION = 36                               # 세대당 생성수
N_BEST = 16                                     # 베스트 수
N_CHILDREN = 7                                  # 자손 유전자 수
PROB_MUTATION = 0.036240651034459274            # 돌연변이
mutation_std = 1.1881981491901645               # 돌연변이시 standard deviation
REVERSE = True                                  # 배열 순서 (False: ascending order, True: descending order) == maximize
score_ini = 0                                   # 초기 점수
process_duration = 32                           # batch size
input_length = (process_duration) * 4 + 1       # neural net의 input length
output_length_1 = 18                            # Event length (CHECK_1~4, PROCESS)
resolution = 3                                  # mol 생산량의 resolution
output_length_2 = (int)(resolution * 6.6 + 2)   # MOL(0~6.6, step:0.1)
h1 = (20,20)                                    # 히든레이어1 노드 수 (event, mol)
h2 = (26,26)                                    # 히든레이어2 노드 수 (event, mol)
h3 = (10,10)                                    # 히든레이어3 노드 수 (event, mol)
EPOCHS = 1000                                   # 반복 횟수
early_stopping = 30                             # saturation시 early stopping
crossover_fraction = 0.6600957352939291         # crossover 비율
save_file_name = 'submit.csv'
#%% Initial guess
genomes = []
for _ in range(N_POPULATION):
    genome = Genome(score_ini, input_length, output_length_1, output_length_2, resolution, h1, h2, h3, process_duration, init_weight=None)
    genomes.append(genome)
try:
    for i in range(N_BEST):
        genomes[i] = best_genomes[i]
except:
    best_genomes = []
    for _ in range(N_BEST): # genome의 개수 여기서 조정
        genome = Genome(score_ini, input_length, output_length_1, output_length_2, resolution, h1, h2, h3, process_duration,init_weight=None)
        best_genomes.append(genome)
print('==Process 1 Done==')

def crossover(process_1, process_2, new_process, weight, crossover_fraction):
    # crossover event
    for j in range(getattr(new_process.event_net, weight).shape[0]):
        w_e = getattr(new_process.event_net, weight).shape[1]
        cut = np.zeros(w_e)
        cut[np.random.choice(range(w_e), (int)(np.floor(w_e * crossover_fraction)))] = 1
        getattr(new_process.event_net, weight)[j, cut == 1] = getattr(process_1.event_net, weight)[j, cut == 1]
        getattr(new_process.event_net, weight)[j, cut == 0] = getattr(process_2.event_net, weight)[j, cut == 0]
    # crossover mol
    for j in range(getattr(new_process.mol_net, weight).shape[0]):
        w_m = getattr(new_process.mol_net, weight).shape[1]
        cut = np.zeros(w_m)
        cut[np.random.choice(range(w_m), (int)(np.floor(w_m * crossover_fraction)))] = 1
        # crossover event
        getattr(new_process.mol_net, weight)[j, cut == 1] = getattr(process_1.mol_net, weight)[j, cut == 1]
        getattr(new_process.mol_net, weight)[j, cut == 0] = getattr(process_2.mol_net, weight)[j, cut == 0]
    return new_process

def mutation(new_process, mean, stddev, weight):
    # mutation event
    w_e = getattr(new_process.event_net, weight).shape[0]
    h_e = getattr(new_process.event_net, weight).shape[1]
    if np.random.uniform(0, 1) < PROB_MUTATION:
        new_process.event_net.__dict__[weight] = new_process.event_net.__dict__[weight] * np.random.normal(mean, stddev, size=(w_e, h_e)) * np.random.randint(0, 2, (w_e, h_e))
    # mutation mol
    w_m = getattr(new_process.mol_net, weight).shape[0]
    h_m = getattr(new_process.mol_net, weight).shape[1]
    if np.random.uniform(0, 1) < PROB_MUTATION:
        new_process.mol_net.__dict__[weight] = new_process.mol_net.__dict__[weight] * np.random.normal(mean, stddev, size=(w_m, h_m)) * np.random.randint(0, 2, (w_m, h_m))
    return new_process

#%% 모델 학습
n_gen = 1 
score_history = []
high_score_history = []
mean_score_history = []
while n_gen <= EPOCHS:
    genomes = np.array(genomes)    
    while len(genomes)%CPU_CORE != 0:
        genomes = np.append(genomes, Genome(score_ini, input_length, output_length_1, output_length_2,resolution, h1, h2, h3, process_duration,init_weight=None))
    genomes = genomes.reshape((len(genomes)//CPU_CORE, CPU_CORE))
    
    for idx, _genomes in enumerate(genomes):
        if __name__ == '__main__':
            pool = multiprocessing.Pool(processes=CPU_CORE)
            genomes[idx] = pool.map(genome_score, _genomes)
            pool.close()
            pool.join()
    genomes = list(genomes.reshape(genomes.shape[0]*genomes.shape[1]))    
    
     # score에 따라 정렬
    genomes.sort(key=lambda x: x.score, reverse=REVERSE)
    
    # 평균 점수
    s = 0 
    for i in range(N_BEST):
        s += genomes[i].score
    s /= N_BEST
    
    # Best Score
    bs = genomes[0].score 
    
    # Best Model 추가
    if best_genomes is not None:
        genomes.extend(best_genomes)
        
    # score에 따라 정렬
    genomes.sort(key=lambda x: x.score, reverse=REVERSE)
    
    score_history.append([n_gen, genomes[0].score])
    high_score_history.append([n_gen, bs])
    mean_score_history.append([n_gen, s])
    
    # 결과 출력
    print('EPOCH #%s\tHistory Best Score: %s\tBest Score: %s\tMean Score: %s' % (n_gen, genomes[0].score, bs, s))    
    
    # 모델 업데이트
    best_genomes = deepcopy(genomes[:N_BEST])

    # CHILDREN 생성 -- 부모로부터 crossover을 해서 children 생성
    for i in range(N_CHILDREN):
        new_genome = deepcopy(best_genomes[0])
        genome_1 = np.random.choice(best_genomes)
        genome_2 = np.random.choice(best_genomes)

        for weight in ['w1','w2','w3','w4','b1','b2','b3','b4']:
            new_genome.process_1.__dict__[weight] = crossover(genome_1.process_1, genome_2.process_1, new_genome.process_1, weight, crossover_fraction)
            new_genome.process_2.__dict__[weight] = crossover(genome_1.process_2, genome_2.process_2, new_genome.process_2, weight, crossover_fraction)
    
    # 모델 초기화
    genomes = []
    for i in range(int(N_POPULATION / len(best_genomes))):
        for bg in best_genomes:
            new_genome = deepcopy(bg)            
            mean = 0
            stddev = mutation_std
            # Mutation
            for weight in ['w1','w2','w3','w4','b1','b2','b3','b4']:
                new_genome.process_1.__dict__[weight] = mutation(new_genome.process_1, mean, stddev, weight)
                new_genome.process_2.__dict__[weight] = mutation(new_genome.process_2, mean, stddev, weight)
            genomes.append(new_genome)
            
    if REVERSE:
        if bs < score_ini:
            genomes[len(genomes)//2:] = [Genome(score_ini, input_length, output_length_1, output_length_2,resolution, h1, h2, h3, process_duration,init_weight=None) for _ in range(N_POPULATION//2)]
    else:
        if bs > score_ini:
            genomes[len(genomes)//2:] = [Genome(score_ini, input_length, output_length_1, output_length_2,resolution, h1, h2, h3, process_duration,init_weight=None) for _ in range(N_POPULATION//2)]

    # early stopping
    if n_gen > early_stopping:
        last_scores = high_score_history[-1 * early_stopping:]
        sub_scores = list(map(lambda x: x[1], last_scores))
        if np.argmax(sub_scores) == 0:
            print('No improvement, early stopping...')
            break

    n_gen += 1

print('==Process 2 Done==')
#%% 결과
import matplotlib.pyplot as plt

# Score Graph
score_history = np.array(score_history)
high_score_history = np.array(high_score_history)
mean_score_history = np.array(mean_score_history)

plt.plot(score_history[:,0], score_history[:,1], '-o', label='BEST')
plt.plot(high_score_history[:,0], high_score_history[:,1], '-o', label='High')
plt.plot(mean_score_history[:,0], mean_score_history[:,1], '-o', label='Mean')
plt.legend()
plt.xlim(0, EPOCHS)
plt.ylim(bottom=0)
plt.xlabel('Epochs')
plt.ylabel('Score')

plt.show()
# 재고 계산
from module.simulator import Simulator
simulator = Simulator()
submission = best_genomes[0].submission
_, df_stock = simulator.get_score(submission) # 재고 계산 (MOL을 만들고 남은 PRT, 음수인 경우 부족한만큼 채워줘야 함)

#%% 제출
# PRT 개수 계산 -- MOL을 바탕으로
PRTs = df_stock[['PRT_1', 'PRT_2', 'PRT_3', 'PRT_4']].values
PRTs = (PRTs[:-1] - PRTs[1:])[24*23:] # 23일 후부터
PRTs = np.ceil(PRTs * 1.1) # 넉넉하게 PRT를 만듦
PAD = np.zeros((24*23+1, 4)) # 마지막 23일은 생산을 안 함
PRTs = np.append(PRTs, PAD, axis=0).astype(int)

# Submission 파일에 PRT 입력
submission.loc[:, 'PRT_1':'PRT_4'] = PRTs
submission.to_csv('submit/'+save_file_name, index=False)