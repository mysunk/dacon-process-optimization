import os
import pandas as pd
import numpy as np
from pathlib import Path

change_time = {'CHANGE_12':6, 'CHANGE_13':13,'CHANGE_14':13,'CHANGE_21':6,'CHANGE_23':13,'CHANGE_24':13,
                    'CHANGE_31':13,'CHANGE_32':13,'CHANGE_34':6,'CHANGE_41':13,'CHANGE_42':13,'CHANGE_43':6}

class Simulator:
    def __init__(self):
        self.sample_submission = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'sample_submission.csv'))
        self.max_count = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'max_count.csv'))
        self.stock = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'stock.csv'))
        order = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'order.csv'), index_col=0)
        order.index = pd.to_datetime(order.index)
        self.order = order
        
    def get_state(self, data):
        if 'CHECK' in data:
            return int(data[-1])
        elif 'CHANGE' in data:
            return int(data[-1])
        else:
            return np.nan
        
    def cal_schedule_part_1(self, df): # 원형 제작 공정
        columns = ['PRT_1', 'PRT_2', 'PRT_3', 'PRT_4']
        df_set = df[columns]
        df_out = df_set * 0
        
        p = 0.985
        dt = pd.Timedelta(days=23)
        end_time = df_out.index[-1]

        for time in df_out.index:
            out_time = time + dt
            if end_time < out_time:
                break
            else:            
                for column in columns:
                    set_num = df_set.loc[time, column]
                    if set_num > 0:
                        out_num = np.sum(np.random.choice(2, set_num, p=[1-p, p]))         
                        df_out.loc[out_time, column] = out_num

        df_out['MOL_1'] = 0.0
        df_out['MOL_2'] = 0.0
        df_out['MOL_3'] = 0.0
        df_out['MOL_4'] = 0.0
        df_out['BLK_1'] = 0.0
        df_out['BLK_2'] = 0.0
        df_out['BLK_3'] = 0.0
        df_out['BLK_4'] = 0.0
        return df_out    
    
    def cal_schedule_part_2(self, df, line='A'): # 성형 공정
        if line == 'A':
            columns = ['Event_A', 'MOL_A']
        elif line == 'B':
            columns = ['Event_B', 'MOL_B']
        else:
            columns = ['Event_A', 'MOL_A']
            
        schedule = df[columns].copy()
        
        schedule['state'] = 0
        schedule['state'] = schedule[columns[0]].apply(lambda x: self.get_state(x))
        schedule['state'] = schedule['state'].fillna(method='ffill')
        schedule['state'] = schedule['state'].fillna(0)
        
        schedule_process = schedule.loc[schedule[columns[0]]=='PROCESS']
        df_out = schedule.drop(schedule.columns, axis=1)
        df_out['PRT_1'] = 0.0
        df_out['PRT_2'] = 0.0
        df_out['PRT_3'] = 0.0
        df_out['PRT_4'] = 0.0
        df_out['MOL_1'] = 0.0
        df_out['MOL_2'] = 0.0
        df_out['MOL_3'] = 0.0
        df_out['MOL_4'] = 0.0

        p = 0.975
        times = schedule_process.index
        for i, time in enumerate(times):
            value = schedule.loc[time, columns[1]]
            state = int(schedule.loc[time, 'state'])
            df_out.loc[time, 'PRT_'+str(state)] = -value
            if i+48 < len(times):
                out_time = times[i+48]
                df_out.loc[out_time, 'MOL_'+str(state)] = value * p

        df_out['BLK_1'] = 0.0
        df_out['BLK_2'] = 0.0
        df_out['BLK_3'] = 0.0
        df_out['BLK_4'] = 0.0
        return df_out

    def cal_stock(self, df, df_order): # 수요가 발생한 시점에 자르기
        df_stock = df * 0

        blk2mol = {}
        blk2mol['BLK_1'] = 'MOL_1'
        blk2mol['BLK_2'] = 'MOL_2'
        blk2mol['BLK_3'] = 'MOL_3'
        blk2mol['BLK_4'] = 'MOL_4'

        cut = {}
        cut['BLK_1'] = 506
        cut['BLK_2'] = 506
        cut['BLK_3'] = 400
        cut['BLK_4'] = 400

        p = {}
        p['BLK_1'] = 0.851
        p['BLK_2'] = 0.901
        blk_diffs = []
        for i, time in enumerate(df.index):
            month = time.month
            if month == 4:
                p['BLK_3'] = 0.710
                p['BLK_4'] = 0.700        
            elif month == 5:
                p['BLK_3'] = 0.742
                p['BLK_4'] = 0.732
            elif month == 6:
                p['BLK_3'] = 0.759
                p['BLK_4'] = 0.749
            else:
                p['BLK_3'] = 0.0
                p['BLK_4'] = 0.0
                
            if i == 0:
                df_stock.iloc[i] = df.iloc[i]    
            else:
                df_stock.iloc[i] = df_stock.iloc[i-1] + df.iloc[i]
                for column in df_order.columns:
                    val = df_order.loc[time, column]
                    if val > 0:
                        mol_col = blk2mol[column]
                        mol_num = df_stock.loc[time, mol_col]
                        df_stock.loc[time, mol_col] = 0     
                        
                        blk_gen = int(mol_num*p[column]*cut[column])
                        blk_stock = df_stock.loc[time, column] + blk_gen
                        blk_diff = blk_stock - val
                        
                        df_stock.loc[time, column] = blk_diff
                        blk_diffs.append(blk_diff)
        return df_stock, blk_diffs

    def subprocess(self, df):
        out = df.copy()
        column = 'time'
        out.index = pd.to_datetime(out[column])
        out = out.drop([column], axis=1)
        out.index.name = column
        return out
    
    def add_stock(self, df, df_stock):
        df_out = df.copy()
        for column in df_out.columns:
            df_out.iloc[0][column] = df_out.iloc[0][column] + df_stock.iloc[0][column]
        return df_out

    def order_rescale(self, df, df_order):
        df_rescale = df.drop(df.columns, axis=1)
        dt = pd.Timedelta(hours=18)
        for column in ['BLK_1', 'BLK_2', 'BLK_3', 'BLK_4']:
            for time in df_order.index:
                df_rescale.loc[time+dt, column] = df_order.loc[time, column]
        df_rescale = df_rescale.fillna(0)
        return df_rescale

    def F(self, x, a):
        if x < a:
            return 1-x/a
        else:
            return 0

    def cal_score(self, blk_diffs, df): # df: sample_submission
        # Block Order Difference
        p = 0 # 생산 부족분 합계
        q = 0 # 생산 초과분 합계
        for item in blk_diffs:
            if item < 0: # 수요 부족분에 대해 더해줌
                p = p + abs(item)
            if item > 0: # 수요 초과분에 대해 더해줌
                q = q + abs(item)
        N = np.sum(self.order.loc[:,'BLK_1':'BLK_4'].values) # 블럭장난감 총 수요
        M = df.shape[0] # 총 시간
        
        # 성형 공정 change 시간 합계
        c, c_n, s, s_n = 0, 0, 0, 0        
        for event_type in ['Event_A', 'Event_B']:
            change_event = [string for string in df[event_type].values if 'CHANGE' in string]
            c = c + len(change_event)
            # 변경 이벤트 횟수
            event , count = np.unique(change_event,return_counts = True)
            for i,e in enumerate(event):
                count[i] = np.ceil(count[i] / change_time[e])
            c_n = c_n + np.sum(count)
            
            # stop 시간 합계
            stop_event = df[event_type].values == 'STOP'
            s = s + sum(stop_event)
            # stop 이벤트 횟수
            for i in range(len(stop_event)-1):
                if stop_event[i] == False and stop_event[i+1] == True:
                    s_n = s_n + 1
        score = 50 * self.F(x=p, a=10*N) + 20 * self.F(q, 10*N) + 20 * self.F(c, M) / (1+0.1 * c_n) + 10 * self.F(s, M) / (1+0.1 * s_n)
        return score
    
    def get_score(self, df):
        df = self.subprocess(df) 
        out_1 = self.cal_schedule_part_1(df)
        out_2 = self.cal_schedule_part_2(df, line='A')
        out_3 = self.cal_schedule_part_2(df, line='B')
        out = out_1 + out_2 + out_3
        out = self.add_stock(out, self.stock)
        order = self.order_rescale(out, self.order)                    
        out, blk_diffs = self.cal_stock(out, order)                    
        score = self.cal_score(blk_diffs, df)
        return score, out