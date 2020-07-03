import os
import pandas as pd
import numpy as np
from pathlib import Path
from module.simulator import Simulator
# from util import *
from scipy.special import expit
import scipy.special as sc

simulator = Simulator()
submission_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'sample_submission.csv'))
order_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'order.csv'))
order_ini = order_ini.loc[:,'BLK_1':'BLK_4'].values
max_count = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'max_count.csv'), dtype={'date': str, 'count': float})
max_count = max_count['count'].values

class network():

    def __init__(self, input_len, output_len, hidden_layer1, hidden_layer2, hidden_layer3):
        # initialize weight
        self.w1 = np.random.randn(input_len, hidden_layer1) * np.sqrt(2 / (input_len + hidden_layer1))
        self.w2 = np.random.randn(hidden_layer1, hidden_layer2) * np.sqrt(2 / (hidden_layer1 + hidden_layer2))
        self.w3 = np.random.randn(hidden_layer2, hidden_layer3) * np.sqrt(2 / (hidden_layer2 + hidden_layer3))
        self.w4 = np.random.randn(hidden_layer3, output_len) * np.sqrt(2 / (hidden_layer3 + output_len))
        self.b1 = np.random.randn(1, hidden_layer1) * np.sqrt(1 / hidden_layer1)
        self.b2 = np.random.randn(1, hidden_layer2) * np.sqrt(1 / hidden_layer2)
        self.b3 = np.random.randn(1, hidden_layer3) * np.sqrt(1 / hidden_layer3)
        self.b4 = np.random.randn(1, output_len) * np.sqrt(1 / output_len)

    def sigmoid(self, x):
        return expit(x)

    def softmax(self, x):
        return np.exp(x - sc.logsumexp(x)) # np.exp(x) / np.sum(np.exp(x), axis=1)

    def linear(self, x):
        return x

    def forward(self, inputs):
        net = np.matmul(inputs, self.w1) + self.b1
        net = self.linear(net)
        net = np.matmul(net, self.w2) + self.b2
        net = self.linear(net)
        net = np.matmul(net, self.w3) + self.b3
        net = (net - np.mean(net)) / np.std(net)  # batch normalization
        net = self.sigmoid(net)
        net = np.matmul(net, self.w4) + self.b4
        net = (net - np.mean(net)) / np.std(net)  # batch normalization
        score = self.softmax(net)
        return score


class process():

    def __init__(self, input_len, output_len_1, output_len_2, resolution, h1, h2, h3, init_weight=None):
        # neural net architecture
        self.resolution = resolution
        self.change_time = [6, 13, 13, 6, 13, 13, 13, 13, 6, 13, 13, 6]

        # Mol and Event network
        self.event_net = network(input_len, output_len_1, h1[0], h2[0], h3[0])
        self.mol_net = network(input_len, output_len_2, h1[1], h2[1], h3[1])
        self.event = submission_ini['Event_A'].copy().values
        self.mol =  submission_ini['MOL_A'].copy().values.astype('float16')

        # Event masking
        self.event_map = {0: 'CHECK_1', 1: 'CHECK_2', 2: 'CHECK_3', 3: 'CHECK_4',
                          4: 'CHANGE_12', 5: 'CHANGE_13', 6: 'CHANGE_14',
                          7: 'CHANGE_21', 8: 'CHANGE_23', 9: 'CHANGE_24',
                          10: 'CHANGE_31', 11: 'CHANGE_32', 12: 'CHANGE_34',
                          13: 'CHANGE_41', 14: 'CHANGE_42', 15: 'CHANGE_43',
                          16: 'PROCESS', 17: 'STOP'}

        self.mask = np.zeros([len(self.event_map)], np.bool)  # Event_map masking

        # process 변수들을 초기화
        self.initialize()

    def update_mask(self):
        self.mask[:] = False
        # 생산중
        # 가능한 이벤트: check, stop, change, process
        if self.process:
            # check and stop
            if self.process_time >= 98:
                # process가 98시간이 넘어가면 check와 stop으로 변경 가능
                self.mask[:4] = True
                self.mask[17] = True
            # change
            if self.change_mode == -1:  # change가 막 끝난 경우는 통과
                self.change_mode = 0
            else:
                for mode_candidate in range(self.process_mode * 3, self.process_mode * 3 + 3):  # change candidates
                    if self.process_time <= 140 - self.change_time[mode_candidate] + 1:
                        self.mask[mode_candidate + 4] = True
            # process
            self.mask[16] = True
        # 생산중이 아님
        # 가능한 이벤트: check, stop, change, process
        else:
            # check - 막 시작
            if self.check_time == 28:
                self.mask[:4] = True
            # stop
            elif self.check_time == -1:
                if self.stop_time < 28:
                    self.mask[17] = True  # stop만 true
                elif self.stop_time < 8 * 24:  # 28시간 초과 8일 미만 == check도 가능
                    self.mask[17] = True
                    self.mask[:4] = True
                else:
                    #  check만 true
                    self.mask[:4] = True
            # check - 진행중
            elif self.check_time < 28:
                if self.change_mode in [0, -1]:  # change 중인지 확인
                    self.mask[self.process_mode] = True  # 해야할 check에 True
                else:  # change
                    self.mask[self.change_mode + 3] = True  # change를 유지

    def update(self, inputs):
        # Event 신경망
        net = self.event_net.forward(inputs)
        net += 1
        net = net * self.mask
        net[np.isnan(net)] = 0
        net[np.isinf(net)] = 0
        out1 = self.event_map[np.argmax(net)]

        # MOL 수량 신경망
        if out1 == 'PROCESS':
            net = self.mol_net.forward(inputs)
            out2 = np.argmax(net)
            out2 /= self.resolution
        else:
            out2 = 0
        return out1, out2

    def check_event(self, num):
        # process -> check
        if self.process == 1:
            self.process = 0
            self.check_time = 28
        # stop -> check
        if self.check_time == -1:
            self.stop_time = 0
            self.check_time = 28
        self.process_mode = num
        self.check_time -= 1
        if self.check_time == 0:
            self.process = 1
            self.process_time = 0

    def change_event(self, out, change_mode):
        self.change_mode = change_mode
        change_to = int(out[-1])
        if self.process == 1:
            self.process = 0
            self.check_time = self.change_time[self.change_mode - 1]
        self.process_time += 1
        self.check_time -= 1
        if self.check_time == 0:
            self.process = 1
            self.change_mode = -1
            self.process_mode = change_to - 1
        # IF 140 exceeds
        if self.process_time >= 140:
            self.process = 0
            self.check_time = 28
            self.change_mode = 0

    def progress_event(self, s, out1, out2):
        # event_map 에서 out1에 해당하는 key return
        even = next((k for k, v in self.event_map.items() if v == out1), None)
        ### Check events
        if 'CHECK' in out1:
            self.check_event(even)
        ### Change events
        elif 'CHANGE' in out1:
            self.change_event(out1, even-3)
        ### Stop event
        elif out1 == 'STOP':
            if self.process == 1:
                self.process = 0
                self.stop_time = 0  # stop event 시작
                self.check_time = -1  # check를 끔
            self.stop_time += 1
            if self.stop_time >= 8 * 24:  # 8일이 넘어갈 경우
                self.stop_time = 0  # stop을 끔
                self.check_time = 28  # check를 켬
        ### Process
        elif out1 == 'PROCESS':
            self.process_time += 1
            if self.process_time >= 140:
                self.process = 0
                self.check_time = 28
                self.change_mode = 0
                self.stop_time = 0
        self.event[s] = out1
        self.mol[s] = out2  # 몇 개를 성형할지

    def initialize(self):
        self.check_time = 28  # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.stop_time = 0  # stop시간 체크, 29 ~ 192까지 가능
        self.change_mode = 0  # event_map의 4~15가 change_mode의 1~12에 대응, 0은 change중이 아님
        self.process = 0  # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode = 0  # 생산 물품 번호 1~4 가 0~3, stop시 4
        self.process_time = 0  # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140


class Genome():

    def __init__(self, score_ini, input_len, output_len_1, output_len_2, resolution, h1, h2, h3, process_duration,
                 init_weight=None):
        self.process_1 = process(input_len, output_len_1, output_len_2, resolution, h1, h2, h3, init_weight=init_weight)
        self.process_2 = process(input_len, output_len_1, output_len_2, resolution, h1, h2, h3, init_weight=init_weight)
        self.process_duration = process_duration
        self.submission = submission_ini

        # 평가 점수 초기화
        self.score = score_ini

        # PRT 처음 재고
        self.stock_PRT = (np.array([0, 258, 0, 0]) * 0.985).astype('float16')
        self.mol_cumsum = np.array([0, 0, 0, 0]).astype('float16')

    def predict(self, order):
        order = np.concatenate([order, np.zeros((self.process_duration, 4)).astype(int)], axis=0)
        self.submission.loc[:, 'PRT_1':'PRT_4'] = 0
        for s in range(self.submission.shape[0]):
            # update mask
            self.process_1.update_mask()
            self.process_2.update_mask()
            if s < 24 * 20:
                # Check event일 때 PRT가 있는 물품만 검사
                if self.process_1.mask[0:4].sum():
                    self.process_1.mask[0:4] = False
                    for i, PRT in enumerate(self.stock_PRT):
                        if PRT > 0: self.process_1.mask[i] = True
                if self.process_2.mask[0:4].sum():
                    self.process_2.mask[0:4] = False
                    for i, PRT in enumerate(self.stock_PRT):
                        if PRT > 0: self.process_2.mask[i] = True
            inputs = np.array(order[s // 24:(s // 24 + self.process_duration),:]).reshape(-1)
            inputs = np.append(inputs, s % 24)

            # update
            event_a, mol_a = self.process_1.update(inputs)
            event_b, mol_b = self.process_2.update(inputs)

            # 6.66보다 많이 생산하면 cut
            if mol_a > 6.666: mol_a = 6.666
            if mol_b > 6.666: mol_b = 6.666

            # max_count를 넘으면 cut
            if self.process_1.mol[s-(s%24):s].sum() + mol_a > max_count[s//24]:
                mol_a = 0
            if self.process_2.mol[s-(s%24):s].sum() + mol_b > max_count[s//24]:
                mol_b = 0

            # If exceeds initial PRT, cut
            if s < 24 * 23:
                if event_a == 'PROCESS':
                    if self.stock_PRT[self.process_1.process_mode] == 0:
                        mol_a = 0
                    else:
                        if self.mol_cumsum[self.process_1.process_mode] + mol_a > self.stock_PRT[self.process_1.process_mode]:
                            mol_a = 0
                        else:
                            self.mol_cumsum[self.process_1.process_mode] += mol_a
                if event_b == 'PROCESS':
                    if self.stock_PRT[self.process_2.process_mode] == 0:
                        mol_b = 0
                    else:
                        if self.mol_cumsum[self.process_2.process_mode] + mol_b > self.stock_PRT[self.process_2.process_mode]:
                            mol_b = 0
                        else:
                            self.mol_cumsum[self.process_2.process_mode] += mol_b

            # update state
            self.process_1.progress_event(s, event_a, mol_a)
            self.process_2.progress_event(s, event_b, mol_b)

        # submission에 넣음
        self.submission['Event_A'], self.submission['Event_B'] = self.process_1.event, self.process_2.event
        self.submission['MOL_A'], self.submission['MOL_B'] = self.process_1.mol, self.process_2.mol

        # 변수 초기화
        self.process_1.initialize()
        self.process_2.initialize()
        self.mol_cumsum = np.array([0, 0, 0, 0]).astype('float16')
        self.process_1.event = submission_ini['Event_A'].copy().values
        self.process_1.event = submission_ini['Event_A'].copy().values
        self.process_1.mol = submission_ini['MOL_A'].copy().values.astype('float16')
        self.process_1.mol = submission_ini['MOL_A'].copy().values.astype('float16')

        return self.submission


def genome_score(genome):
    submission = genome.predict(order_ini)
    genome.score, _ = simulator.get_score(submission)
    return genome
