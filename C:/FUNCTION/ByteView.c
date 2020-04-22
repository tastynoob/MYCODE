//#include"ByteView.h"
#include<stdio.h>
#include<Windows.h>
#define MaxColumn 20//最大显示列数



FILE* fp;

int main(int argc, char* argv[]) {
	if (argc != 2) {
		printf("#参数错误\n");
		return -1;
	}
	fp = fopen(argv[1],"rb+");
	if (!fp) {
		printf("#文件不存在\n");
		return -1;
	}
	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	int counter = 0;
	for (int a; (a = fgetc(fp)) != EOF; counter++) {
		if (counter == MaxColumn) {
			counter = 0;
			printf("\n");
		}
		if (a < 17)printf("0");
		printf("%x,", a);
	}
	printf("\n#文件长度:%d\n", len);
	printf("#文件显示完毕");
	
}
