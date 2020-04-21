#include<math.h>
#include"BPNetWork.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

//神经网络的层数
#define LS network->layernum

//神经网络隐藏层与输出层的层数
#define AS network->layernum-1

//输入层神经元的数量
#define INNS network->neurons[0]

//输入层的第a个输入
#define INS(a) network->input[a-1]

//输出层神经元的数量
#define OUTNS network->neurons[network->layernum-1]

//第n层神经元的数量
#define NS(n) network->neurons[n-1]


//第n层第a个神经元的第p个权重
#define WF(n,a,p) network->layers[n-2].ws[(p-1) + (a-1)*network->neurons[n-2]]

//第n层的第a个神经元的偏置
#define BF(n,a) network->layers[n-2].bs[a-1]

//第n层第a个神经元的输出
#define OF(n,a) network->layers[n-2].ps[a-1]

//第n层第a个神经元的误差
#define SF(n,a) network->layers[n-2].ss[a-1]

BPNetWork* BPCreate(int* nums, int len,double ln)
{
	BPNetWork* network = malloc(sizeof(BPNetWork));
	network->layernum = len;
	network->neurons = malloc(len * sizeof(int));
	network->ln = ln;
	memcpy(network->neurons, nums, len * sizeof(int));
	//
	network->input = malloc(nums[0] * sizeof(double));
	network->layers = malloc(sizeof(Layer) * (len - 1));
	network->out = malloc(sizeof(double) * nums[len - 1]);
	for (int p = 0; p < len - 1; p++) {
		srand(&__TIME__);
		int lastnum = nums[p];//上一层的神经元数量
		int num = nums[p + 1];//当前层的神经元数量
		network->layers[p].nums = num;
		network->layers[p].bs = malloc(sizeof(double) * num);
		//
		network->layers[p].ws = malloc(sizeof(double) * num * lastnum);
		//
		network->layers[p].ps = malloc(sizeof(double) * num);
		//
		network->layers[p].ss = malloc(sizeof(double) * num);
		for (int pp = 0; pp < num; pp++) {
			network->layers[p].bs[pp] = rand() / RAND_MAX;
			for (int ppp = 0; ppp < lastnum; ppp++) {
				network->layers[p].ws[ppp + pp * lastnum] = rand() / RAND_MAX;
			}
		}
	}
	return network;
}
//运行一次神经网络
void RunOnce(BPNetWork* network) {
	//计算输入层到第二层
	for (int a = 1; a <= NS(2); a++) {
		double net = 0;
		double* putout = &OF(2,a);//获取第2层的输出值
		*putout = 0;
		for (int aa = 1; aa <= INNS; aa++) {
			net += INS(aa) * WF(2, a, aa);
		}
		*putout = f(net + BF(2,a));
	}
	for (int n = 2; n <= LS-1; n++) {
		for (int a = 1; a <= NS(n + 1); a++) {//下一层的神经网络
			double net = 0;
			double* putout = &OF(n+1,a);
			*putout = 0;
			for (int aa = 1; aa <= NS(n); aa++) {//当前层的神经网络
				net += OF(n, aa) * WF(n + 1, a, aa);
			}
			*putout = f(net + BF(n + 1, a));
		}
	}
}

//载入训练集
void LoadIn(BPNetWork* network,double* input,double* putout) {
	memcpy(network->input, input, INNS*sizeof(double));
	memcpy(network->out, putout, OUTNS*sizeof(double));
}

//反向传播一次(训练一次)
void TrainOnce(BPNetWork* network) {
	//计算输出层的误差函数
	for (int a = 1; a <= OUTNS; a++) {
		double* s = &SF(LS,a);//获取第a个神经元的误差
		*s = 0;
		double* b = &BF(LS, a);//获取第a个神经元的偏置
		double o = OF(LS, a);//获取第a个神经元的输出
		*s = 2 * (o - network->out[a]) * f_(o);
		*b = *b - network->ln * (*s);//更新偏置
		//更新权重
		for (int aa = 1; aa <=NS(LS-1) ; aa++) {
			double* w = &WF(LS, a, aa);
			*w = *w - network->ln * (*s) * OF(LS - 1, aa);
		}
	}
	//计算隐藏层的误差
	for (int a = LS-1; a > 2; a--) {
		//开始计算第a层每个神经元的误差
		for (int n = 1; n <= NS(a); n++) {//当前层
			double* s = &SF(a, n);//获取第a个神经元的误差
			*s = 0;
			double* b = &BF(a, n);//获取第a个神经元的偏置
			double o = OF(a, n);//获取第a个神经元的输出
			for (int nn = 1; nn <= NS(a+1); nn++) {//下一层
				double lw = WF(a + 1, nn, n);//获取下一层到当前神经元的偏置
				double ls = SF(a + 1, nn);//获取下一层第nn个神经元的误差
				*s += ls * lw * f_(o);
			}
			*b = *b - network->ln * (*s);//更新偏置
			//更新权重
			for (int nn = 1; nn <= NS(a - 1); nn++) {//上一层
				double* w = &WF(a, n, nn);
				*w = *w - network->ln * (*s) *OF(a - 1, nn);
			}
		}
	}
	//计算第2层的误差函数
	for (int n = 1; n <= NS(2); n++) {//当前层
		double* s = &SF(2, n);//获取第a个神经元的误差
		*s = 0;
		double* b = &BF(2, n);//获取第a个神经元的偏置
		double o = OF(2, n);//获取第a个神经元的输出
		for (int nn = 1; nn <= NS(3); nn++) {//下一层
			double lw = WF(3, nn, n);//获取下一层到当前神经元的偏置
			double ls = SF(3, nn);//获取下一层第nn个神经元的误差
			*s += ls * lw * f_(o);
		}
		*b = *b - network->ln * (*s);//更新偏置
		//更新权重
		for (int nn = 1; nn <= INNS; nn++) {//上一层
			double* w = &WF(2, n, nn);
			*w = *w - network->ln * (*s) * INS(nn);
		}
	}
}
//输出总误差
double ETotal(BPNetWork* network) {
	double val = 0;
	for (int a = 1; a <= OUTNS; a++) {
		val += (OF(network->layernum, a) - network->out[a]) * (OF(network->layernum, a) - network->out[a]) / OUTNS;
	}
	return val;
}

int main() {
	int a[] = { 5,100,100,5 };
	double in[5] = { 10,10,10,10,10, };
	double out[5] = { 0.1,0.1,0.1,0.1,0.1 };


	double in1[5] = { 3,3,3,3,3, };
	double out1[5] = { 0.5,0.1,0.3,0.1,0.2 };
	BPNetWork* network = BPCreate(a, 4,0.5);
	
	

	while (1)
	{
		int w = 100;
		LoadIn(network, in, out);
		while (w--)
		{
			RunOnce(network);
			TrainOnce(network);
			printf("%g\n", ETotal(network));


			//system("cls");
		}
		w = 100;
		printf("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW");
		LoadIn(network, in1, out1);
		while (w--)
		{
			RunOnce(network);
			TrainOnce(network);
			printf("%g\n", ETotal(network));
		}
	}
	
	
	return 0;
}
















