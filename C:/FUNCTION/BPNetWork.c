#include<math.h>
#include"BPNetWork.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

//神经网络的层数
#define LS network->lns

//输入层神经元的数量
#define INNS network->ns[0]

//输入层的第a个输入
#define INS(a) network->is[a-1]

//第a个理想输出
#define TAS(a) network->ts[a-1]

//输出层神经元的数量
#define OUTNS network->ns[LS-1]

//第n层神经元的数量
#define NS(n) network->ns[n-1]

//第n层第a个神经元的第p个权重
#define WF(n,a,p) network->las[n-2].ws[(p-1)+(a-1)*NS(n-1)]

//第n层的第a个神经元的偏置
#define BF(n,a) network->las[n-2].bs[a-1]

//第n层第a个神经元的输出
#define OF(n,a) network->las[n-2].os[a-1]

//第n层第a个神经元的误差
#define SF(n,a) network->las[n-2].ss[a-1]

//学习率
#define LN network->ln

BPNetWork* BPCreate(int* nums, int len,double ln)
{
	BPNetWork* network = malloc(sizeof(BPNetWork));
	network->lns = len;
	network->ns = malloc(len * sizeof(int));
	network->ln = ln;
	memcpy(network->ns, nums, len * sizeof(int));
	//
	network->is = malloc(nums[0] * sizeof(double));
	network->las = malloc(sizeof(Layer) * (len - 1));
	network->ts = malloc(sizeof(double) * nums[len - 1]);
	for (int p = 0; p < len - 1; p++) {
		srand(&p);
		int lastnum = nums[p];//上一层的神经元数量
		int num = nums[p + 1];//当前层的神经元数量
		network->las[p].bs = malloc(sizeof(double) * num);
		//
		network->las[p].ws = malloc(sizeof(double) * num * lastnum);
		//
		network->las[p].os = malloc(sizeof(double) * num);
		//
		network->las[p].ss = malloc(sizeof(double) * num);
		for (int pp = 0; pp < num; pp++) {
			network->las[p].bs[pp] = 0.2;//rand() / 2.0 / RAND_MAX;
			for (int ppp = 0; ppp < lastnum; ppp++) {
				network->las[p].ws[ppp + pp * lastnum] = 0.5;//rand() / 2.0 / RAND_MAX;
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
		double* o = &OF(2,a);//获取第2层的输出值
		*o = 0;
		for (int aa = 1; aa <= INNS; aa++) {
			net += INS(aa) * WF(2, a, aa);//INS(aa) * WF(2, a, aa);
		}
		*o = f(net + BF(2,a));
	}
	for (int n = 2; n <= LS-1; n++) {
		for (int a = 1; a <= NS(n + 1); a++) {//下一层的神经网络
			double net = 0;
			double* o = &OF(n+1,a);
			*o = 0;
			for (int aa = 1; aa <= NS(n); aa++) {//当前层的神经网络
				double oo = OF(n, aa);
				double* ww = &WF(n + 1, a, aa);
				net += oo * (*ww);
			}
			*o = f(net + BF(n + 1, a));
			*o = *o;
		}
	}
}

//载入训练集
void LoadIn(BPNetWork* network,double* input,double* putout) {
	memcpy(network->is, input, INNS*sizeof(double));
	memcpy(network->ts, putout, OUTNS*sizeof(double));
}

//反向传播一次(训练一次)
void TrainOnce(BPNetWork* network) {
	//计算输出层的误差函数
	for (int a = 1; a <= OUTNS; a++) {
		double* s = &SF(LS,a);//获取第a个神经元的误差
		*s = 0;
		double* b = &BF(LS, a);//获取第a个神经元的偏置
		double o = OF(LS, a);//获取第a个神经元的输出
		*s = (2.0 / OUTNS) * (o - TAS(a))* f_(o);
		*b = *b - LN * (*s);//更新偏置
		//更新权重
		for (int aa = 1; aa <=NS(LS-1) ; aa++) {
			double* w = &WF(LS, a, aa);
			*w = *w - LN * (*s) * OF(LS-1, aa);
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
			*b = *b - LN * (*s);//更新偏置
			//更新权重
			for (int nn = 1; nn <= NS(a - 1); nn++) {//上一层
				double* w = &WF(a, n, nn);
				*w = *w - LN * (*s) *OF(a - 1, nn);
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
		*b = *b - LN * (*s);//更新偏置
		//更新权重
		for (int nn = 1; nn <= INNS; nn++) {//上一层
			double* w = &WF(2, n, nn);
			*w = *w - LN * (*s) * INS(nn);
		}
	}
	
}
//输出总误差
double ETotal(BPNetWork* network) {
	double val = 0;
	for (int a = 1; a <= OUTNS; a++) {
		val += ((OF(LS, a) - TAS(a)) * (OF(LS, a) - TAS(a))) / OUTNS;
	}
	return val;
}

int main() {
	int a[] = { 1,20,20,1 };
	double in[1] = { 0.5 };
	double in1[1] = { 0.2 };


	double out[1] = { 0.1 };



	BPNetWork* network = BPCreate(a, 4, 0.5);

	double* l = &BF(3, 1);

	LoadIn(network, in, out);

	while (1)
	{
		RunOnce(network);
		TrainOnce(network);
		double a = OF(4, 1);
		printf("%g\n", a);
	}






	return 0;
}
















