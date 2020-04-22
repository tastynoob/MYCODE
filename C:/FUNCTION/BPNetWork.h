#ifndef BPNETWORK_H
#define BPNETWORK_H
//所需头文件
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>


#define f(x) Sigmoid(x)//激活函数设定
#define f_(x) Sigmoidf(x)//导函数

typedef struct {
	double* ws;//权重矩阵
	double* bs;//偏置数组
	double* os;//输出数组
	double* ss;//误差(总误差关于加权和的偏导)
} Layer;
typedef struct {
	int lns;//层数
	int* ns;//每层神经元的数量
	double* is;//神经网络输入
	double* ts;//理想输出
	Layer* las;//神经网络各个层(不包括输入层)
	double ln;//学习率
}BPNetWork;








//创建神经网络
BPNetWork* BPCreate(int* nums, int len,double ln);
//运行一次神经网络
void RunOnce(BPNetWork* network);
//载入训练集
void LoadIn(BPNetWork* network, double* input, double* putout);
//反向传播一次(训练一次)
void TrainOnce(BPNetWork* network);
//输出总误差
double ETotal(BPNetWork* network);




//sigmoid激活函数
#define Sigmoid(x)  (1 / (1 + exp(-(x))))
//sigmoid激活函数的导函数,输入为sigmoid输出
#define Sigmoidf(f)  ((f) * (1 - (f)))


#define Tanh(x) ((2 / (1 + exp(-2 * (x))))-1)
#define Tanhf(f) ((1+(f))*(1-(f)))




#endif
