#ifndef BPNETWORK_H
#define BPNETWORK_H

#define f(x) Sigmoid(x)//激活函数设定
#define f_(x) Sigmoidf(x)//导函数

typedef struct {
	int nums;//神经元的数量
	double* ws;//权重矩阵
	double* bs;//偏置数组
	double* ps;//输出数组
	double* ss;//误差(总误差关于加权和的偏导)
} Layer;
typedef struct {
	int layernum;//层数
	int* neurons;//每层神经元的数量
	double* input;//神经网络输入
	double* out;//理想输出
	Layer* layers;//神经网络各个层(不包括输入层)
	double ln;//学习率
}BPNetWork;








//创建神经网络
BPNetWork* BPCreate(int* nums, int len,double ln);
//运行一次神经网络
void RunOnce(BPNetWork* network);





//sigmoid激活函数
#define Sigmoid(x) 1 / (1 + exp(-x))
//sigmoid激活函数的导函数,输入为sigmoid输出
#define Sigmoidf(f) f * (1 - f) 
#define Tanh(x) (2 / (1 + exp(-2 * x)))-1
#define Tanhf(f) 2*(1+f)*(1-f)




#endif
