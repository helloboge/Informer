import argparse
import os
import torch
import pandas as pd
from PyEMD import CEEMDAN
import numpy as np
from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=False, default='london_merged', help='data')
# parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
#
# parser.add_argument('--data', type=str, default='WTH', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
# parser.add_argument('--root_path', type=str, default='/kaggle/working/Informer2020/data/ETT/', help='root path of the data file')

parser.add_argument('--data_path', type=str, required=False, default='london_merged.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# parser.add_argument('--checkpoints', type=str, default='/kaggle/working/Informer2020/checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_false', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='exp',help='test exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()


def ceemdan_decompose(series=None, trials=3):
    decom = CEEMDAN()
    decom.trials = trials
    df_ceemdan = pd.DataFrame(decom(series.values).T)
    df_ceemdan.columns = ['imf'+str(i) for i in range(len(df_ceemdan.columns))]
    return df_ceemdan


def predict_with_ceemdan_and_informer(ceemdan_result):
    final_prediction = np.zeros((len(ceemdan_result), args.pred_len))
    final_trues = np.zeros((len(ceemdan_result), args.pred_len))
    for i in range(len(ceemdan_result.columns)):
        # 获取当前分量的数据
        current_data = ceemdan_result['imf' + str(i)]
        folder_path = './data/ETT/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # 创建一个 DataFrame 包含当前分量数据

        df_raw_data['cnt'] = current_data
        df = pd.DataFrame({'imf{}'.format(i): df_raw_data})
        # 保存为CSV文件，文件名为 imf0.csv, imf1.csv, ...
        filename = 'london_imf{}.csv'.format(i)
        df.to_csv(filename, index=False)

        print('Saved {} to {}'.format(df.columns[0], filename))
        # 使用Informer模型进行训练和预测
        # 这里需要根据你的实际情况调用你的Informer模型进行训练和预测
        # 以下是一个示例，实际需要根据你的模型接口进行调用
        exp,metrics,preds,trues = train_and_predict_with_informer(current_data,args)

        # 将当前分量的预测结果累加到最终预测结果上
        final_prediction += preds
        final_trues += trues

    return exp,metrics,final_prediction,final_trues


def train_and_predict_with_informer(current_data,args):
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.data = 'current_data'
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # data：数据文件名，T：标签列，M：预测变量数(如果要预测12个特征，则为[12,12,12]),
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
        'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
        'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
        'Tianchi_power': {'data': 'Tianchi_power.csv', 'T': 'power_consumption', 'M': [2, 2, 2], 'S': [1, 1, 1],
                          'MS': [2, 2, 1]},
        'rainning': {'data': 'rainning.csv', 'T': 'rainnum', 'M': [5, 5, 5], 'S': [1, 1, 1], 'MS': [5, 5, 1]},
        'london_merged': {'data': 'london_merged.csv', 'T': 'cnt', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
        'london_imf0': {'data': 'london_merged.csv', 'T': 'cnt', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
        'london_imf1': {'data': 'london_merged.csv', 'T': 'cnt', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
        'london_imf2': {'data': 'london_merged.csv', 'T': 'cnt', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
        'london_imf3': {'data': 'london_merged.csv', 'T': 'cnt', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},

    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    print('实验参数:')
    print(args)

    Exp = Exp_Informer

    for ii in range(args.itr):
        # 设置实验记录
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.attn,
            args.factor,
            args.embed,
            args.distil,
            args.mix,
            args.des,
            ii)

        exp = Exp(args)  # 设置实验
        print('>>>>>>>开始训练 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        exp.train(setting)

        print('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        metrics,preds,trues = exp.test(setting)

        if args.do_predict:
            print('>>>>>>>预测 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            real_preds = exp.predict(setting, True)

        torch.cuda.empty_cache()

    return exp,metrics,preds,trues

if __name__ == '__main__':
    # 加载数据和预处理
    df_raw_data = pd.read_csv('./data/ETT/london_merged.csv', header=0, parse_dates=['date'],
                              date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d %H:%M'))
    series_close = pd.Series(df_raw_data['cnt'].values,index = df_raw_data['date'])
    print(series_close)

    # 调用 CEEMDAN 分解函数
    ceemdan_result = ceemdan_decompose(series_close)  # series_close是你的分解前数据
    print(ceemdan_result)
    # 使用 Informer 模型进行训练和预测
    exp,metrics,final_prediction,final_trues = predict_with_ceemdan_and_informer(ceemdan_result)

    print(final_prediction.shape)
    print(final_trues.shape)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(final_trues[0, :, -1], label='GroundTruth')
    plt.plot(final_prediction[0, :, -1], label='Prediction')
    plt.legend()
    plt.show()
    # 展示结果或保存结果
    # show_results(final_prediction)  # 这里需要根据你的实际情况实现
    # save_results(final_prediction)  # 这里需要根据你的实际情况实现
'''

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=False, default='london_merged', help='data')
# parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
#
# parser.add_argument('--data', type=str, default='WTH', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
# parser.add_argument('--root_path', type=str, default='/kaggle/working/Informer2020/data/ETT/', help='root path of the data file')

parser.add_argument('--data_path', type=str, required=False, default='london_merged.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# parser.add_argument('--checkpoints', type=str, default='/kaggle/working/Informer2020/checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_false', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='exp',help='test exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# data：数据文件名，T：标签列，M：预测变量数(如果要预测12个特征，则为[12,12,12]),
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'Tianchi_power': {'data': 'Tianchi_power.csv', 'T': 'power_consumption', 'M': [2, 2, 2], 'S': [1, 1, 1], 'MS': [2, 2, 1]},
    'rainning': {'data': 'rainning.csv', 'T': 'rainnum', 'M': [5, 5, 5], 'S': [1, 1, 1],'MS': [5, 5, 1]},
    'london_merged':{'data':'london_merged.csv','T':'cnt','M':[6,6,6],'S':[1,1,1],'MS':[6,6,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)

    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    metrics,preds,trues=exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        real_preds = exp.predict(setting, True)

    torch.cuda.empty_cache()
'''