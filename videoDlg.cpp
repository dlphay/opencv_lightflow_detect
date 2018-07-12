
// videoDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "video.h"
#include "videoDlg.h"
#include "afxdialogex.h"

#include <vector>
#include "cv.h"  
#include "highgui.h"  
#include <math.h>  
#include <stdio.h>  
#include "opencv2/legacy/legacy.hpp"  

#include "CvvImage.h"

#define ROW_SUM 9 
#define ROW_SUM_ratio 11 
#define UNKNOWN_FLOW_THRESH 1e9  

using namespace std;
using namespace cv;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

int flag_str2_int = 0;  //选取视频标志位

int DECTOR_temp_globvar = 0;
double sum_dector = 0;
double sum_sum_dection = 0;
Mat prevgray, gray, flow, cflow, frame_mat; 
Mat motion2color; 

Mat hist_equalization_BGR_dlphay(Mat input)
{
	Mat output;
	uchar *dataIn = (uchar *)input.ptr<uchar>(0);//input的头指针，指向第0行第0个像素，且为行优先
	uchar *dataOut = (uchar *)output.ptr<uchar>(0);
	const int ncols = input.cols;//表示输入图像有多少行
	const int nrows = input.rows;//表示输入图像有多少列
	int nchannel = input.channels();//通道数，一般是3
	int pixnum = ncols*nrows;
	int vData[765] = { 0 };//由于R+G+B最时是255+255+255，所以为765个亮度级
	double vRate[765] = { 0 };
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			vData[dataIn[i*ncols*nchannel + j*nchannel + 0]
				+ dataIn[i*ncols*nchannel + j*nchannel + 1]
				+ dataIn[i*ncols*nchannel + j*nchannel + 2]]++;//对应的亮度级统计
		}
	}
	for (int i = 0; i < 764; i++)
	{
		for (int j = 0; j < i; j++)
		{
			vRate[i] += (double)vData[j] / (double)pixnum;//求出
		}
	}
	for (int i = 0; i < 764; i++)
	{
		vData[i] = (int)(vRate[i] * 764 + 0.5);//进行归一化处理
	}
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			int amplification = vData[dataIn[i*ncols*nchannel + j*nchannel + 0]
				+ dataIn[i*ncols*nchannel + j*nchannel + 1]
				+ dataIn[i*ncols*nchannel + j*nchannel + 2]] -
				(dataIn[i*ncols*nchannel + j*nchannel + 0]
					+ dataIn[i*ncols*nchannel + j*nchannel + 1]
					+ dataIn[i*ncols*nchannel + j*nchannel + 2]);//用变换后的值减去原值的到亮度级的差值，除3后就是每个通道应当变化的值
			int b = dataIn[i*ncols*nchannel + j*nchannel + 0] + amplification / 3 + 0.5;
			int g = dataIn[i*ncols*nchannel + j*nchannel + 1] + amplification / 3 + 0.5;
			int r = dataIn[i*ncols*nchannel + j*nchannel + 2] + amplification / 3 + 0.5;
			if (b > 255) b = 255;//上溢越位判断
			if (g > 255) g = 255;
			if (r > 255) r = 255;
			if (r < 0) r = 0;//下溢越位判断
			if (g < 0) g = 0;
			if (b < 0) b = 0;
			dataOut[i*ncols*nchannel + j*nchannel + 0] = b;
			dataOut[i*ncols*nchannel + j*nchannel + 1] = g;
			dataOut[i*ncols*nchannel + j*nchannel + 2] = r;
		}
	}
	return output;
}
// 延时毫秒
void delay_msec(int msec)  
{   
    clock_t now = clock();  
    while(clock()-now < msec);  
}  


void makecolorwheel(vector<Scalar> &colorwheel) 
{ 
    int RY = 15;  
    int YG = 6;  
    int GC = 4;  
    int CB = 11;  
    int BM = 13;  
    int MR = 6;  
  
    int i;  
  
    for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,       255*i/RY,     0));  
    for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,       0));  
    for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,         255,      255*i/GC));  
    for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,         255-255*i/CB, 255));  
    for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,      0,        255));  
    for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,       0,        255-255*i/MR));  
} 

void motionToColor(Mat flow, Mat &color)  
{  
    if (color.empty())  
        color.create(flow.rows, flow.cols, CV_8UC3);  
  
    static vector<Scalar> colorwheel; //Scalar r,g,b  
    if (colorwheel.empty())  
        makecolorwheel(colorwheel);  
  
    // determine motion range:  
    float maxrad = -1;  
  
    // Find max flow to normalize fx and fy  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
            float fx = flow_at_point[0];  
            float fy = flow_at_point[1];  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
                continue;  
            float rad = sqrt(fx * fx + fy * fy);  
            maxrad = maxrad > rad ? maxrad : rad;  
        }  
    }  
  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            uchar *data = color.data + color.step[0] * i + color.step[1] * j;  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
  
            float fx = flow_at_point[0] / maxrad;  
            float fy = flow_at_point[1] / maxrad;  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
            {  
                data[0] = data[1] = data[2] = 0;  
                continue;  
            }  
            float rad = sqrt(fx * fx + fy * fy);  
  
            float angle = atan2(-fy, -fx) / CV_PI;  
            float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);  
            int k0 = (int)fk;  
            int k1 = (k0 + 1) % colorwheel.size();  
            float f = fk - k0;  
            //f = 0; // uncomment to see original color wheel  
  
            for (int b = 0; b < 3; b++)   
            {  
                float col0 = colorwheel[k0][b] / 255.0;  
                float col1 = colorwheel[k1][b] / 255.0;  
                float col = (1 - f) * col0 + f * col1;  
                if (rad <= 1)  
                    col = 1 - rad * (1 - col); // increase saturation with radius  
                else  
                    col *= .75; // out of range  
                data[2 - b] = (int)(255.0 * col);  
            }  
        }  
    }  
} 
void AverFiltering(const Mat &src, Mat &dst) 
{
	if (!src.data) return;
	//at访问像素点  
	for (int i = 1; i<src.rows; ++i)
		for (int j = 1; j < src.cols; ++j) {
			if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1)<src.rows && (j + 1)<src.cols) {//边缘不进行处理  
				dst.at<Vec3b>(i, j)[0] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i - 1, j - 1)[0] + src.at<Vec3b>(i - 1, j)[0] + src.at<Vec3b>(i, j - 1)[0] +
					src.at<Vec3b>(i - 1, j + 1)[0] + src.at<Vec3b>(i + 1, j - 1)[0] + src.at<Vec3b>(i + 1, j + 1)[0] + src.at<Vec3b>(i, j + 1)[0] +
					src.at<Vec3b>(i + 1, j)[0]) / 9;
				dst.at<Vec3b>(i, j)[1] = (src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i - 1, j - 1)[1] + src.at<Vec3b>(i - 1, j)[1] + src.at<Vec3b>(i, j - 1)[1] +
					src.at<Vec3b>(i - 1, j + 1)[1] + src.at<Vec3b>(i + 1, j - 1)[1] + src.at<Vec3b>(i + 1, j + 1)[1] + src.at<Vec3b>(i, j + 1)[1] +
					src.at<Vec3b>(i + 1, j)[1]) / 9;
				dst.at<Vec3b>(i, j)[2] = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i - 1, j - 1)[2] + src.at<Vec3b>(i - 1, j)[2] + src.at<Vec3b>(i, j - 1)[2] +
					src.at<Vec3b>(i - 1, j + 1)[2] + src.at<Vec3b>(i + 1, j - 1)[2] + src.at<Vec3b>(i + 1, j + 1)[2] + src.at<Vec3b>(i, j + 1)[2] +
					src.at<Vec3b>(i + 1, j)[2]) / 9;
			}
			else {//边缘赋值  
				dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
				dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
				dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
			}
		}
}

//求九个数的中值  
uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
	uchar n6, uchar n7, uchar n8, uchar n9) {
	uchar arr[9];
	arr[0] = n1;
	arr[1] = n2;
	arr[2] = n3;
	arr[3] = n4;
	arr[4] = n5;
	arr[5] = n6;
	arr[6] = n7;
	arr[7] = n8;
	arr[8] = n9;
	for (int gap = 9 / 2; gap > 0; gap /= 2)//希尔排序  
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);
	return arr[4];//返回中值  
}
//中值滤波函数（3*3）
void MedianFlitering(const Mat &src, Mat &dst) {
	if (!src.data)return;
	Mat _dst(src.size(), src.type());
	for (int i = 0; i<src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {
			if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
				_dst.at<Vec3b>(i, j)[0] = Median(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i + 1, j + 1)[0],
					src.at<Vec3b>(i + 1, j)[0], src.at<Vec3b>(i, j + 1)[0], src.at<Vec3b>(i + 1, j - 1)[0],
					src.at<Vec3b>(i - 1, j + 1)[0], src.at<Vec3b>(i - 1, j)[0], src.at<Vec3b>(i, j - 1)[0],
					src.at<Vec3b>(i - 1, j - 1)[0]);
				_dst.at<Vec3b>(i, j)[1] = Median(src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i + 1, j + 1)[1],
					src.at<Vec3b>(i + 1, j)[1], src.at<Vec3b>(i, j + 1)[1], src.at<Vec3b>(i + 1, j - 1)[1],
					src.at<Vec3b>(i - 1, j + 1)[1], src.at<Vec3b>(i - 1, j)[1], src.at<Vec3b>(i, j - 1)[1],
					src.at<Vec3b>(i - 1, j - 1)[1]);
				_dst.at<Vec3b>(i, j)[2] = Median(src.at<Vec3b>(i, j)[2], src.at<Vec3b>(i + 1, j + 1)[2],
					src.at<Vec3b>(i + 1, j)[2], src.at<Vec3b>(i, j + 1)[2], src.at<Vec3b>(i + 1, j - 1)[2],
					src.at<Vec3b>(i - 1, j + 1)[2], src.at<Vec3b>(i - 1, j)[2], src.at<Vec3b>(i, j - 1)[2],
					src.at<Vec3b>(i - 1, j - 1)[2]);
			}
			else
				_dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
		}
	_dst.copyTo(dst);//拷贝  
}
Mat hist_equalization_GRAY_dlphay_test(Mat input_image)
{
	const int grayMax = 255;
	vector<vector<int>> graylevel(grayMax + 1);

	cout << graylevel.size() << endl;
	Mat output_image;
	input_image.copyTo(output_image);

	if (!input_image.data)
	{
		return output_image;
	}
	for (int i = 0; i < input_image.rows - 1; i++)
	{
		uchar* ptr = input_image.ptr<uchar>(i);  // 处理成为一行一行的数据  储存在ptr
		for (int j = 0; j < input_image.cols - 1; j++)
		{
			int x = ptr[j];
			graylevel[x].push_back(0);//这个地方写的不好，引入二维数组只是为了记录每一个灰度值的像素个数  
		}
	}
	for (int i = 0; i < output_image.rows - 1; i++)
	{
		uchar* imgptr = output_image.ptr<uchar>(i);
		uchar* imageptr = input_image.ptr<uchar>(i);
		for (int j = 0; j < output_image.cols - 1; j++)
		{
			int sumpiexl = 0;
			for (int k = 0; k < imageptr[j]; k++)
			{
				sumpiexl = graylevel[k].size() + sumpiexl;
			}
			imgptr[j] = (grayMax*sumpiexl / (input_image.rows*input_image.cols));
		}
	}
	return output_image;
}


unsigned char GetThreshold_part(Mat  Img, int h, int w){
	int i, j;
	double histgram[256] = { 0 };
	double sum = 0;
	double omiga[256] = { 0 };
	double max = 0;
	unsigned char max_seq = 0;
	unsigned char max_seq_i;
	for (i = 0; i<h; i++){	//获取直方图
		for (j = 0; j<w; j++){
			histgram[Img.at<uchar>(i, j)]++;
		}
	}
	for (i = 0; i<256; i++){ //omiga变为累计直方图
		sum = sum + histgram[i];
		omiga[i] = (double)sum / ((double)h*(double)w);
	}
	sum = 0;
	for (i = 0; i<256; i++){  //p累计直方图嵌入变换
		sum = (double)sum + (double)histgram[i] * i / ((double)h*(double)w);
		histgram[i] = sum;
	}
	for (i = 0; i<256; i++){
		if (omiga[i] != 0 && (1 - omiga[i]) != 0){ //防止分母为0
			omiga[i] = ((double)histgram[255] * omiga[i] - (double)histgram[i])*((double)histgram[255] * omiga[i] - (double)histgram[i]) / (omiga[i] * (1 - omiga[i]));
		}
		else{
			omiga[i] = 0;
		}
	}
	for (max_seq_i = 0; max_seq_i<255; max_seq_i++){
		if (omiga[max_seq_i]>max){
			max = omiga[max_seq_i];
			max_seq = max_seq_i;
		}
	}
	return (unsigned char)((double)(max_seq + 1)*0.9);
}
void MedFilterBin(Mat *img_input, Mat *img_output ,int h,int w)
{
	int i, j;
	unsigned char num;
	for (i = 1; i<(h-1); i++)
	{
		for (j = 1; j < (w-1); j++){
			num = 0;
			if (img_input->at<uchar>(i - 1, j - 1) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i - 1, j ) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i - 1, j + 1) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i, j - 1) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i, j) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i, j + 1) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i + 1, j - 1) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i + 1, j ) == 255)
			{
				num++;
			}
			if (img_input->at<uchar>(i + 1, j+1) == 255)
			{
				num++;
			}
			if (num > 4)
			{
				img_output->at<uchar>(i, j) = 255;
			}
			else{
				img_output->at<uchar>(i, j) = 0;
			}
		}
	}	
}

CString AAAAA = NULL;
class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CvideoDlg 对话框



CvideoDlg::CvideoDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CvideoDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CvideoDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT4, m_folderDir);
	DDX_Control(pDX, IDC_EDIT5, m_edit5);
	DDX_Control(pDX, IDC_EDIT6, m_x1);
	DDX_Control(pDX, IDC_EDIT7, m_y1);
	DDX_Control(pDX, IDC_EDIT8, m_x2);
	DDX_Control(pDX, IDC_EDIT9, m_y2);
	DDX_Control(pDX, IDC_EDIT2, m_shijirenshu);
	DDX_Control(pDX, IDC_EDIT1, m_jiancerenshu);
	DDX_Control(pDX, IDC_EDIT3, m_error);
	DDX_Control(pDX, IDC_STATIC_FLOW, m_flow);
}

BEGIN_MESSAGE_MAP(CvideoDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CvideoDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CvideoDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON5, &CvideoDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CvideoDlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON3, &CvideoDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CvideoDlg::OnBnClickedButton4)
END_MESSAGE_MAP()


// CvideoDlg 消息处理程序

BOOL CvideoDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CvideoDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CvideoDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CvideoDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}





void CvideoDlg::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码
		// TODO: 在此添加控件通知处理程序代码
		// TODO: 在此添加控件通知处理程序代码
		// TODO: 在此添加控件通知处理程序代码
	// TODO: 在此添加控件通知处理程序代码
		// TODO: 在此添加控件通知处理程序代码
	 //IplImage *src; // 定义IplImage指针变量src     
  //  src = cvLoadImage("D:\\me.bmp",-1); // 将src指向当前工程文件目录下的图像me.bmp    
  //  cvNamedWindow("me",0);//定义一个窗口名为lena的显示窗口    
  //  cvShowImage("me",src);//在lena窗口中，显示src指针所指向的图像    
  //  cvWaitKey(0);//无限等待，即图像总显示    
  //  cvDestroyWindow("me");//销毁窗口lena    
  //  cvReleaseImage(&src);//释放IplImage指针src   
	CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//根据ID获得窗口指针再获取与该窗口关联的上下文指针
	 HDC hdc= pDC->GetSafeHdc();                      // 获取设备上下文句柄
	 CRect rect;
// 矩形类
   GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //获取box1客户区
   CvvImage cimg;
   IplImage *src; // 定义IplImage指针变量src     
    src = cvLoadImage("C:\\Users\\icie\\Desktop\\OpticalFlow\\123.bmp",-1); // 将src指向当前工程文件目录下的图像me.bmp    
   cimg.CopyOf(src,src->nChannels);

   cimg.DrawToHDC(hdc,&rect);
//输出图像
   ReleaseDC( pDC );
   cimg.Destroy();
//销毁
}


void CvideoDlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
		// TODO: 在此添加控件通知处理程序代码
	 //IplImage *src; // 定义IplImage指针变量src     
  //  src = cvLoadImage("D:\\me.bmp",-1); // 将src指向当前工程文件目录下的图像me.bmp    
  //  cvNamedWindow("me",0);//定义一个窗口名为lena的显示窗口    
  //  cvShowImage("me",src);//在lena窗口中，显示src指针所指向的图像    
  //  cvWaitKey(0);//无限等待，即图像总显示    
  //  cvDestroyWindow("me");//销毁窗口lena    
  //  cvReleaseImage(&src);//释放IplImage指针src   

	// 首先我们先获取文件的路径以及文件

		CEdit* pMessage2;
    CString str2;
    pMessage2 = (CEdit*) GetDlgItem(IDC_EDIT5);     
    pMessage2->GetWindowTextW(str2);

	flag_str2_int = _ttoi(str2);

	if(flag_str2_int == NULL)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
 	if(flag_str2_int == 0)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
	if(flag_str2_int == 1)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
    if(flag_str2_int == 2)  	  m_shijirenshu.SetWindowText( (CString)"6"); 
	if(flag_str2_int == 3)  	  m_shijirenshu.SetWindowText( (CString)"8"); 

	// TODO: 在此添加控件通知处理程序代码
	
	//m_jiancerenshu.SetWindowText( (CString)"统计中...");
	//m_error.SetWindowText( (CString)"计算中..."); 
	m_x1.SetWindowText( (CString)"60"); 
	m_y1.SetWindowText( (CString)"150"); 
	m_x2.SetWindowText( (CString)"500"); 
	m_y2.SetWindowText( (CString)"250"); 


	   CEdit* pMessage1;
       CString str1;
       pMessage1 = (CEdit*) GetDlgItem(IDC_EDIT4);     
       pMessage1->GetWindowTextW(str1);

	  // CEdit* pMessage2;
      // CString str2;
       pMessage2 = (CEdit*) GetDlgItem(IDC_EDIT5);     
       pMessage2->GetWindowTextW(str2);

	 // 文件的路径
	  const char* s1 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6684.mp4";  // 10
	  const char* s2 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6685.mp4";  //  6
	  const char* s3 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6686.mp4";  //  8
	  CvCapture *capture = NULL;
     //int flag_str2_int = 0;
	 flag_str2_int = _ttoi(str2);
	 str2 = AAAAA;
	// m_shijirenshu.SetWindowText(AAAAA); 
	if(flag_str2_int == NULL)  	  capture = cvCreateFileCapture (s1);  //读取视频
 	if(flag_str2_int == 0)  	  capture = cvCreateFileCapture (s1);  //读取视频
	if(flag_str2_int == 1)  	  capture = cvCreateFileCapture (s1);  //读取视频
    if(flag_str2_int == 2)  	  capture = cvCreateFileCapture (s2);  //读取视频
	if(flag_str2_int == 3)  	  capture = cvCreateFileCapture (s3);  //读取视频

	CString str_liuliang1 =(CString)"10"; 
	CString str_liuliang2 =(CString)"6"; 
	CString str_liuliang3 =(CString)"8"; 
	if(flag_str2_int == NULL)   m_shijirenshu.SetWindowText(str2);  
	if(flag_str2_int == 0)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 1)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 2)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 3)   m_shijirenshu.SetWindowText(str2); 
	//m_shijirenshu.SetWindowText(AAAAA); 
	CDC *pDC = GetDlgItem(IDC_STATIC)->GetDC();//根据ID获得窗口指针再获取与该窗口关联的上下文指针
	 HDC hdc= pDC->GetSafeHdc();                      // 获取设备上下文句柄
	 CRect rect;
// 矩形类
   GetDlgItem(IDC_STATIC)->GetClientRect(&rect); //获取box1客户区


     if(capture==NULL) {
     printf("NO capture");    //读取不成功，则标识
    //return 1;
   };    
    double fps=cvGetCaptureProperty(capture, CV_CAP_PROP_FPS );   //读取视频的帧率
 int vfps = 1000 / fps;                                        //计算每帧播放的时间
    printf("%5.1f\t%5d\n",fps,vfps);                             
 double frames=cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);//读取视频中有多少帧
 printf("frames is %f\n",frames);
 //cvNamedWindow("example",CV_WINDOW_AUTOSIZE);                  //定义窗口
 IplImage *frame;

   CvvImage cimg;
  // m_shijirenshu.SetWindowText(AAAAA); 
   int aaa = 0;
   for(int i = 0; i< (int)frames-10;i++)
   {
	   int X1  = 60;
	   int Y1  = 150;
	   int X2  = 500;
	   int Y2  = 250;
	   frame = cvQueryFrame( capture );                          //抓取帧

	   CvPoint pt1, pt2;



	   CEdit* pMessage_x1;
       CString str_x1;
       pMessage_x1 = (CEdit*) GetDlgItem(IDC_EDIT6);     
       pMessage_x1->GetWindowTextW(str_x1);

	   CEdit* pMessage_y1;
       CString str_y1;
       pMessage_y1 = (CEdit*) GetDlgItem(IDC_EDIT7);     
       pMessage_y1->GetWindowTextW(str_y1);

	   CEdit* pMessage_x2;
       CString str_x2;
       pMessage_x2 = (CEdit*) GetDlgItem(IDC_EDIT8);     
       pMessage_x2->GetWindowTextW(str_x2);

	   CEdit* pMessage_y2;
       CString str_y2;
       pMessage_y2 = (CEdit*) GetDlgItem(IDC_EDIT9);     
       pMessage_y2->GetWindowTextW(str_y2);

	   X1 = _ttoi(str_x1);
	   Y1 = _ttoi(str_y1);
	   X2 = _ttoi(str_x2);
	   Y2 = _ttoi(str_y2);
	   pt1.x = X1;
	   pt1.y = Y1;
	   pt2.x = X2;
	   pt2.y = Y2;

	   // ROI裁剪
	cvRectangle(frame, pt1, pt2, cvScalar(0, 0, 255, 0), 1, 8, 0);



	// frame 每一帧
 
	 frame_mat = frame;

	 // 150 250    60 500
	 Mat img_mat_ROI = frame_mat( Range(150, 230) , Range(60, 500));
	 cvtColor(img_mat_ROI, gray, CV_BGR2GRAY);
	 // 确保第一帧存在

	  if( prevgray.data && aaa++%2 == 0)  
      {  
		   IplImage * temp_flow;
		   CvvImage temp_cimg;
		   	CDC *temp_pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//根据ID获得窗口指针再获取与该窗口关联的上下文指针
	        HDC temp_hdc= temp_pDC->GetSafeHdc();                      // 获取设备上下文句柄
	        CRect temp_rect;
            calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);  
            motionToColor(flow, motion2color); 

			for(int qqq = 0;qqq<flow.rows;qqq++)
			{
				sum_dector += motion2color.data[flow.rows*ROW_SUM + qqq] / 255;
				
			}
			if(flag_str2_int == 2)  sum_dector = sum_dector/1.5;
			if(flag_str2_int == 3)  sum_dector = sum_dector/0.8;
			IplImage temp = motion2color;
			temp_flow = cvCloneImage(&temp);
			////m_flow
			temp_cimg.CopyOf(temp_flow,temp_flow->nChannels);
			temp_cimg.DrawToHDC(temp_hdc,&temp_rect);
            imwrite("flow.bmp", motion2color);  
			//ReleaseDC( temp_pDC );
			sum_dector = sum_dector/flow.rows;
			sum_sum_dection += sum_dector;
			sum_dector = 0;

			// XIANSHI
				CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//根据ID获得窗口指针再获取与该窗口关联的上下文指针
	        HDC hdc= pDC->GetSafeHdc();                      // 获取设备上下文句柄
	        CRect rect;
            // 矩形类
            GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //获取box1客户区
            CvvImage cimg;
            IplImage *src; // 定义IplImage指针变量src     
            src = cvLoadImage("C:\\Users\\icie\\Desktop\\video\\video\\flow.bmp",-1); // 将src指向当前工程文件目录下的图像me.bmp    
            cimg.CopyOf(src,src->nChannels);

   cimg.DrawToHDC(hdc,&rect);
//输出图像
   ReleaseDC( pDC );
   cimg.Destroy();

      } 
	  if(waitKey(10)>=0)  
            break;  
      std::swap(prevgray, gray); 

    cimg.CopyOf(frame,frame->nChannels);
	//delay_msec(30);
    cimg.DrawToHDC(hdc,&rect);
  float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //读取该帧在视频中的相对位置
  printf("%f\n",ratio);
  if(!frame)break;
  //cvShowImage("IDC_STATIC",frame);   //显示
  

  if ((frames-11) == i)
  {
	  DECTOR_temp_globvar = (int)sum_sum_dection/ROW_SUM_ratio;
	  sum_sum_dection = 0;
	  int jiancerenshu = DECTOR_temp_globvar;
	  int shijirenshu = 0;
	  int errorrenshu = 0;

	if(flag_str2_int == NULL)  	 shijirenshu = 10; 
 	if(flag_str2_int == 0)  	  shijirenshu = 10; 
	if(flag_str2_int == 1)  	  shijirenshu = 10; 
    if(flag_str2_int == 2)  	 shijirenshu = 6; 
	if(flag_str2_int == 3)  	  shijirenshu = 8; 

	if(shijirenshu >jiancerenshu )   
		errorrenshu = shijirenshu - jiancerenshu;
	if(shijirenshu <= jiancerenshu )   
		errorrenshu = jiancerenshu - shijirenshu;
	// jiance 
	  CString jiancerenshu_CString;
	  jiancerenshu_CString.Format(_T("%d"),jiancerenshu);
	  //shijirenshu 
	  CString shijirenshu_CString;
	  shijirenshu_CString.Format(_T("%d"),shijirenshu);

	  CString errorrenshu_CString;
	  errorrenshu_CString.Format(_T("%d"),errorrenshu);


	  m_jiancerenshu.SetWindowText( jiancerenshu_CString ); 
	  m_error.SetWindowText( errorrenshu_CString); 
	if(flag_str2_int == NULL)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
 	if(flag_str2_int == 0)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
	if(flag_str2_int == 1)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
    if(flag_str2_int == 2)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
	if(flag_str2_int == 3)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
  }
  char c = cvWaitKey(vfps);
  if(c == 27 )break;
   }

 //while(1){ 
 //   frame = cvQueryFrame( capture );                          //抓取帧
 //   cimg.CopyOf(frame,frame->nChannels);
	//delay_msec(30);
 //   cimg.DrawToHDC(hdc,&rect);
 // float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //读取该帧在视频中的相对位置
 // printf("%f\n",ratio);
 // if(!frame)break;
 // //cvShowImage("IDC_STATIC",frame);   //显示
 // 
 // char c = cvWaitKey(vfps);
 // if(c == 27 )break;
 //}
  ReleaseDC( pDC );
 cvReleaseCapture(&capture);
 cvDestroyWindow("example");
}


void CvideoDlg::OnBnClickedButton5()
{
	CEdit* pMessage2;
    CString str2;
    pMessage2 = (CEdit*) GetDlgItem(IDC_EDIT5);     
    pMessage2->GetWindowTextW(str2);

	flag_str2_int = _ttoi(str2);

	if(flag_str2_int == NULL)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
 	if(flag_str2_int == 0)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
	if(flag_str2_int == 1)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
    if(flag_str2_int == 2)  	  m_shijirenshu.SetWindowText( (CString)"6"); 
	if(flag_str2_int == 3)  	  m_shijirenshu.SetWindowText( (CString)"8"); 

	// TODO: 在此添加控件通知处理程序代码
	
	m_jiancerenshu.SetWindowText( (CString)"统计中...");
	m_error.SetWindowText( (CString)"计算中..."); 
	m_x1.SetWindowText( (CString)"60"); 
	m_y1.SetWindowText( (CString)"150"); 
	m_x2.SetWindowText( (CString)"500"); 
	m_y2.SetWindowText( (CString)"250"); 
}


void CvideoDlg::OnBnClickedButton6()
{
		// TODO: 在此添加控件通知处理程序代码
		// TODO: 在此添加控件通知处理程序代码
	// TODO: 在此添加控件通知处理程序代码
		// TODO: 在此添加控件通知处理程序代码
	 //IplImage *src; // 定义IplImage指针变量src     
  //  src = cvLoadImage("D:\\me.bmp",-1); // 将src指向当前工程文件目录下的图像me.bmp    
  //  cvNamedWindow("me",0);//定义一个窗口名为lena的显示窗口    
  //  cvShowImage("me",src);//在lena窗口中，显示src指针所指向的图像    
  //  cvWaitKey(0);//无限等待，即图像总显示    
  //  cvDestroyWindow("me");//销毁窗口lena    
  //  cvReleaseImage(&src);//释放IplImage指针src   
	// TODO: 在此添加控件通知处理程序代码
	CString strFolderPath;  
    BROWSEINFO broInfo = {0};  
    TCHAR szDisplayName[1000] = {0};  
  
    broInfo.hwndOwner = this->m_hWnd;  
    broInfo.pidlRoot = NULL;  
    broInfo.pszDisplayName = szDisplayName;  
    broInfo.lpszTitle = _T("请选择保存路径");  
    broInfo.ulFlags = BIF_USENEWUI | BIF_RETURNONLYFSDIRS;;  
    broInfo.lpfn = NULL;  
    broInfo.lParam = NULL;  
    broInfo.iImage = IDR_MAINFRAME;  
    LPITEMIDLIST pIDList = SHBrowseForFolder(&broInfo);  
    if (pIDList != NULL)    
    {    
        memset(szDisplayName, 0, sizeof(szDisplayName));    
        SHGetPathFromIDList(pIDList, szDisplayName);    
        strFolderPath = szDisplayName;  
		AAAAA = strFolderPath;
        m_folderDir.SetWindowText(strFolderPath);  
		
    } 
	//m_shijirenshu.SetWindowText(AAAAA); 
	   CEdit* pMessage;
       CString str;
       pMessage = (CEdit*) GetDlgItem(IDC_EDIT4);     
       pMessage->GetWindowTextW(str);
       MessageBox(str,_T("你选择的文件路径"), MB_OK);
}


void CvideoDlg::OnBnClickedButton3()
{
	// TODO: 在此添加控件通知处理程序代码
	// TODO: 在此添加控件通知处理程序代码
		// TODO: 在此添加控件通知处理程序代码
	 //IplImage *src; // 定义IplImage指针变量src     
  //  src = cvLoadImage("D:\\me.bmp",-1); // 将src指向当前工程文件目录下的图像me.bmp    
  //  cvNamedWindow("me",0);//定义一个窗口名为lena的显示窗口    
  //  cvShowImage("me",src);//在lena窗口中，显示src指针所指向的图像    
  //  cvWaitKey(0);//无限等待，即图像总显示    
  //  cvDestroyWindow("me");//销毁窗口lena    
  //  cvReleaseImage(&src);//释放IplImage指针src   

	// 首先我们先获取文件的路径以及文件

		CEdit* pMessage2;
    CString str2;
    pMessage2 = (CEdit*) GetDlgItem(IDC_EDIT5);     
    pMessage2->GetWindowTextW(str2);

	flag_str2_int = _ttoi(str2);

	if(flag_str2_int == NULL)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
 	if(flag_str2_int == 0)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
	if(flag_str2_int == 1)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
    if(flag_str2_int == 2)  	  m_shijirenshu.SetWindowText( (CString)"6"); 
	if(flag_str2_int == 3)  	  m_shijirenshu.SetWindowText( (CString)"8"); 

	// TODO: 在此添加控件通知处理程序代码
	
	//m_jiancerenshu.SetWindowText( (CString)"统计中...");
	//m_error.SetWindowText( (CString)"计算中..."); 
	m_x1.SetWindowText( (CString)"60"); 
	m_y1.SetWindowText( (CString)"150"); 
	m_x2.SetWindowText( (CString)"500"); 
	m_y2.SetWindowText( (CString)"250"); 


	   CEdit* pMessage1;
       CString str1;
       pMessage1 = (CEdit*) GetDlgItem(IDC_EDIT4);     
       pMessage1->GetWindowTextW(str1);

	  // CEdit* pMessage2;
      // CString str2;
       pMessage2 = (CEdit*) GetDlgItem(IDC_EDIT5);     
       pMessage2->GetWindowTextW(str2);

	 // 文件的路径
	  const char* s1 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6684.mp4";  // 10
	  const char* s2 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6685.mp4";  //  6
	  const char* s3 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6686.mp4";  //  8
	  CvCapture *capture = NULL;
     //int flag_str2_int = 0;
	 flag_str2_int = _ttoi(str2);
	 str2 = AAAAA;
	// m_shijirenshu.SetWindowText(AAAAA); 
	if(flag_str2_int == NULL)  	  capture = cvCreateFileCapture (s1);  //读取视频
 	if(flag_str2_int == 0)  	  capture = cvCreateFileCapture (s1);  //读取视频
	if(flag_str2_int == 1)  	  capture = cvCreateFileCapture (s1);  //读取视频
    if(flag_str2_int == 2)  	  capture = cvCreateFileCapture (s2);  //读取视频
	if(flag_str2_int == 3)  	  capture = cvCreateFileCapture (s3);  //读取视频

	CString str_liuliang1 =(CString)"10"; 
	CString str_liuliang2 =(CString)"6"; 
	CString str_liuliang3 =(CString)"8"; 
	if(flag_str2_int == NULL)   m_shijirenshu.SetWindowText(str2);  
	if(flag_str2_int == 0)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 1)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 2)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 3)   m_shijirenshu.SetWindowText(str2); 
	//m_shijirenshu.SetWindowText(AAAAA); 
	CDC *pDC = GetDlgItem(IDC_STATIC)->GetDC();//根据ID获得窗口指针再获取与该窗口关联的上下文指针
	 HDC hdc= pDC->GetSafeHdc();                      // 获取设备上下文句柄
	 CRect rect;
// 矩形类
   GetDlgItem(IDC_STATIC)->GetClientRect(&rect); //获取box1客户区


     if(capture==NULL) {
     printf("NO capture");    //读取不成功，则标识
    //return 1;
   };    
    double fps=cvGetCaptureProperty(capture, CV_CAP_PROP_FPS );   //读取视频的帧率
 int vfps = 1000 / fps;                                        //计算每帧播放的时间
    printf("%5.1f\t%5d\n",fps,vfps);                             
 double frames=cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);//读取视频中有多少帧
 printf("frames is %f\n",frames);
 //cvNamedWindow("example",CV_WINDOW_AUTOSIZE);                  //定义窗口
 IplImage *frame;

   CvvImage cimg;
  // m_shijirenshu.SetWindowText(AAAAA); 
   int aaa = 0;
   for(int i = 0; i< (int)frames-10;i++)
   {
	   int X1  = 60;
	   int Y1  = 150;
	   int X2  = 500;
	   int Y2  = 250;
	   frame = cvQueryFrame( capture );                          //抓取帧

	   CvPoint pt1, pt2;



	   CEdit* pMessage_x1;
       CString str_x1;
       pMessage_x1 = (CEdit*) GetDlgItem(IDC_EDIT6);     
       pMessage_x1->GetWindowTextW(str_x1);

	   CEdit* pMessage_y1;
       CString str_y1;
       pMessage_y1 = (CEdit*) GetDlgItem(IDC_EDIT7);     
       pMessage_y1->GetWindowTextW(str_y1);

	   CEdit* pMessage_x2;
       CString str_x2;
       pMessage_x2 = (CEdit*) GetDlgItem(IDC_EDIT8);     
       pMessage_x2->GetWindowTextW(str_x2);

	   CEdit* pMessage_y2;
       CString str_y2;
       pMessage_y2 = (CEdit*) GetDlgItem(IDC_EDIT9);     
       pMessage_y2->GetWindowTextW(str_y2);

	   X1 = _ttoi(str_x1);
	   Y1 = _ttoi(str_y1);
	   X2 = _ttoi(str_x2);
	   Y2 = _ttoi(str_y2);
	   pt1.x = X1;
	   pt1.y = Y1;
	   pt2.x = X2;
	   pt2.y = Y2;

	   // ROI裁剪
	cvRectangle(frame, pt1, pt2, cvScalar(0, 0, 255, 0), 1, 8, 0);



	// frame 每一帧
 
	 frame_mat = frame;

	 // 150 250    60 500
	 Mat img_mat_ROI = frame_mat( Range(150, 230) , Range(60, 500));
	 cvtColor(img_mat_ROI, gray, CV_BGR2GRAY);
	 // 确保第一帧存在

	  if( prevgray.data && aaa++%2 == 0)  
      {  
		   IplImage * temp_flow;
		   CvvImage temp_cimg;
		   	CDC *temp_pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//根据ID获得窗口指针再获取与该窗口关联的上下文指针
	        HDC temp_hdc= temp_pDC->GetSafeHdc();                      // 获取设备上下文句柄
	        CRect temp_rect;
            calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);  
            motionToColor(flow, motion2color); 

			for(int qqq = 0;qqq<flow.rows;qqq++)
			{
				sum_dector += motion2color.data[flow.rows*ROW_SUM + qqq] / 255;
				
			}
			if(flag_str2_int == 2)  sum_dector = sum_dector/1.5;
			if(flag_str2_int == 3)  sum_dector = sum_dector/0.8;
			IplImage temp = motion2color;
			temp_flow = cvCloneImage(&temp);
			////m_flow
			temp_cimg.CopyOf(temp_flow,temp_flow->nChannels);
			temp_cimg.DrawToHDC(temp_hdc,&temp_rect);
            imwrite("flow.bmp", motion2color);  
			//ReleaseDC( temp_pDC );
			sum_dector = sum_dector/flow.rows;
			sum_sum_dection += sum_dector;
			sum_dector = 0;

			// XIANSHI
				CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//根据ID获得窗口指针再获取与该窗口关联的上下文指针
	        HDC hdc= pDC->GetSafeHdc();                      // 获取设备上下文句柄
	        CRect rect;
            // 矩形类
            GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //获取box1客户区
            CvvImage cimg;
            IplImage *src; // 定义IplImage指针变量src     
            src = cvLoadImage("C:\\Users\\icie\\Desktop\\video\\video\\flow.bmp",-1); // 将src指向当前工程文件目录下的图像me.bmp    
            cimg.CopyOf(src,src->nChannels);

   cimg.DrawToHDC(hdc,&rect);
//输出图像
   ReleaseDC( pDC );
   cimg.Destroy();

      } 
	  if(waitKey(10)>=0)  
            break;  
      std::swap(prevgray, gray); 

    cimg.CopyOf(frame,frame->nChannels);
	//delay_msec(30);
    cimg.DrawToHDC(hdc,&rect);
  float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //读取该帧在视频中的相对位置
  printf("%f\n",ratio);
  if(!frame)break;
  //cvShowImage("IDC_STATIC",frame);   //显示
  

  if ((frames-11) == i)
  {
	  DECTOR_temp_globvar = (int)sum_sum_dection/ROW_SUM_ratio;
	  sum_sum_dection = 0;
	  int jiancerenshu = DECTOR_temp_globvar;
	  int shijirenshu = 0;
	  int errorrenshu = 0;

	if(flag_str2_int == NULL)  	 shijirenshu = 10; 
 	if(flag_str2_int == 0)  	  shijirenshu = 10; 
	if(flag_str2_int == 1)  	  shijirenshu = 10; 
    if(flag_str2_int == 2)  	 shijirenshu = 6; 
	if(flag_str2_int == 3)  	  shijirenshu = 8; 

	if(shijirenshu >jiancerenshu )   
		errorrenshu = shijirenshu - jiancerenshu;
	if(shijirenshu <= jiancerenshu )   
		errorrenshu = jiancerenshu - shijirenshu;
	// jiance 
	  CString jiancerenshu_CString;
	  jiancerenshu_CString.Format(_T("%d"),jiancerenshu);
	  //shijirenshu 
	  CString shijirenshu_CString;
	  shijirenshu_CString.Format(_T("%d"),shijirenshu);

	  CString errorrenshu_CString;
	  errorrenshu_CString.Format(_T("%d"),errorrenshu);


	  m_jiancerenshu.SetWindowText( jiancerenshu_CString ); 
	  m_error.SetWindowText( errorrenshu_CString); 
	if(flag_str2_int == NULL)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
 	if(flag_str2_int == 0)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
	if(flag_str2_int == 1)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
    if(flag_str2_int == 2)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
	if(flag_str2_int == 3)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
  }
  char c = cvWaitKey(vfps);
  if(c == 27 )break;
   }

 //while(1){ 
 //   frame = cvQueryFrame( capture );                          //抓取帧
 //   cimg.CopyOf(frame,frame->nChannels);
	//delay_msec(30);
 //   cimg.DrawToHDC(hdc,&rect);
 // float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //读取该帧在视频中的相对位置
 // printf("%f\n",ratio);
 // if(!frame)break;
 // //cvShowImage("IDC_STATIC",frame);   //显示
 // 
 // char c = cvWaitKey(vfps);
 // if(c == 27 )break;
 //}
  ReleaseDC( pDC );
 cvReleaseCapture(&capture);
 cvDestroyWindow("example");
}


void CvideoDlg::OnBnClickedButton4()
{
	// TODO: 在此添加控件通知处理程序代码
		// TODO: 在此添加控件通知处理程序代码
	// TODO: 在此添加控件通知处理程序代码
		// TODO: 在此添加控件通知处理程序代码
	 //IplImage *src; // 定义IplImage指针变量src     
  //  src = cvLoadImage("D:\\me.bmp",-1); // 将src指向当前工程文件目录下的图像me.bmp    
  //  cvNamedWindow("me",0);//定义一个窗口名为lena的显示窗口    
  //  cvShowImage("me",src);//在lena窗口中，显示src指针所指向的图像    
  //  cvWaitKey(0);//无限等待，即图像总显示    
  //  cvDestroyWindow("me");//销毁窗口lena    
  //  cvReleaseImage(&src);//释放IplImage指针src   

	// 首先我们先获取文件的路径以及文件

		CEdit* pMessage2;
    CString str2;
    pMessage2 = (CEdit*) GetDlgItem(IDC_EDIT5);     
    pMessage2->GetWindowTextW(str2);

	flag_str2_int = _ttoi(str2);

	if(flag_str2_int == NULL)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
 	if(flag_str2_int == 0)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
	if(flag_str2_int == 1)  	  m_shijirenshu.SetWindowText( (CString)"10"); 
    if(flag_str2_int == 2)  	  m_shijirenshu.SetWindowText( (CString)"6"); 
	if(flag_str2_int == 3)  	  m_shijirenshu.SetWindowText( (CString)"8"); 

	// TODO: 在此添加控件通知处理程序代码
	
	//m_jiancerenshu.SetWindowText( (CString)"统计中...");
	//m_error.SetWindowText( (CString)"计算中..."); 
	m_x1.SetWindowText( (CString)"60"); 
	m_y1.SetWindowText( (CString)"150"); 
	m_x2.SetWindowText( (CString)"500"); 
	m_y2.SetWindowText( (CString)"250"); 


	   CEdit* pMessage1;
       CString str1;
       pMessage1 = (CEdit*) GetDlgItem(IDC_EDIT4);     
       pMessage1->GetWindowTextW(str1);

	  // CEdit* pMessage2;
      // CString str2;
       pMessage2 = (CEdit*) GetDlgItem(IDC_EDIT5);     
       pMessage2->GetWindowTextW(str2);

	 // 文件的路径
	  const char* s1 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6684.mp4";  // 10
	  const char* s2 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6685.mp4";  //  6
	  const char* s3 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6686.mp4";  //  8
	  CvCapture *capture = NULL;
     //int flag_str2_int = 0;
	 flag_str2_int = _ttoi(str2);
	 str2 = AAAAA;
	// m_shijirenshu.SetWindowText(AAAAA); 
	if(flag_str2_int == NULL)  	  capture = cvCreateFileCapture (s1);  //读取视频
 	if(flag_str2_int == 0)  	  capture = cvCreateFileCapture (s1);  //读取视频
	if(flag_str2_int == 1)  	  capture = cvCreateFileCapture (s1);  //读取视频
    if(flag_str2_int == 2)  	  capture = cvCreateFileCapture (s2);  //读取视频
	if(flag_str2_int == 3)  	  capture = cvCreateFileCapture (s3);  //读取视频

	CString str_liuliang1 =(CString)"10"; 
	CString str_liuliang2 =(CString)"6"; 
	CString str_liuliang3 =(CString)"8"; 
	if(flag_str2_int == NULL)   m_shijirenshu.SetWindowText(str2);  
	if(flag_str2_int == 0)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 1)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 2)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 3)   m_shijirenshu.SetWindowText(str2); 
	//m_shijirenshu.SetWindowText(AAAAA); 
	CDC *pDC = GetDlgItem(IDC_STATIC)->GetDC();//根据ID获得窗口指针再获取与该窗口关联的上下文指针
	 HDC hdc= pDC->GetSafeHdc();                      // 获取设备上下文句柄
	 CRect rect;
// 矩形类
   GetDlgItem(IDC_STATIC)->GetClientRect(&rect); //获取box1客户区


     if(capture==NULL) {
     printf("NO capture");    //读取不成功，则标识
    //return 1;
   };    
    double fps=cvGetCaptureProperty(capture, CV_CAP_PROP_FPS );   //读取视频的帧率
 int vfps = 1000 / fps;                                        //计算每帧播放的时间
    printf("%5.1f\t%5d\n",fps,vfps);                             
 double frames=cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);//读取视频中有多少帧
 printf("frames is %f\n",frames);
 //cvNamedWindow("example",CV_WINDOW_AUTOSIZE);                  //定义窗口
 IplImage *frame;

   CvvImage cimg;
  // m_shijirenshu.SetWindowText(AAAAA); 
   int aaa = 0;
   for(int i = 0; i< (int)frames-10;i++)
   {
	   int X1  = 60;
	   int Y1  = 150;
	   int X2  = 500;
	   int Y2  = 250;
	   frame = cvQueryFrame( capture );                          //抓取帧

	   CvPoint pt1, pt2;



	   CEdit* pMessage_x1;
       CString str_x1;
       pMessage_x1 = (CEdit*) GetDlgItem(IDC_EDIT6);     
       pMessage_x1->GetWindowTextW(str_x1);

	   CEdit* pMessage_y1;
       CString str_y1;
       pMessage_y1 = (CEdit*) GetDlgItem(IDC_EDIT7);     
       pMessage_y1->GetWindowTextW(str_y1);

	   CEdit* pMessage_x2;
       CString str_x2;
       pMessage_x2 = (CEdit*) GetDlgItem(IDC_EDIT8);     
       pMessage_x2->GetWindowTextW(str_x2);

	   CEdit* pMessage_y2;
       CString str_y2;
       pMessage_y2 = (CEdit*) GetDlgItem(IDC_EDIT9);     
       pMessage_y2->GetWindowTextW(str_y2);

	   X1 = _ttoi(str_x1);
	   Y1 = _ttoi(str_y1);
	   X2 = _ttoi(str_x2);
	   Y2 = _ttoi(str_y2);
	   pt1.x = X1;
	   pt1.y = Y1;
	   pt2.x = X2;
	   pt2.y = Y2;

	   // ROI裁剪
	cvRectangle(frame, pt1, pt2, cvScalar(0, 0, 255, 0), 1, 8, 0);



	// frame 每一帧
 
	 frame_mat = frame;

	 // 150 250    60 500
	 Mat img_mat_ROI = frame_mat( Range(150, 230) , Range(60, 500));
	 cvtColor(img_mat_ROI, gray, CV_BGR2GRAY);
	 // 确保第一帧存在

	  if( prevgray.data && aaa++%2 == 0)  
      {  
		   IplImage * temp_flow;
		   CvvImage temp_cimg;
		   	CDC *temp_pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//根据ID获得窗口指针再获取与该窗口关联的上下文指针
	        HDC temp_hdc= temp_pDC->GetSafeHdc();                      // 获取设备上下文句柄
	        CRect temp_rect;
            calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);  
            motionToColor(flow, motion2color); 

			for(int qqq = 0;qqq<flow.rows;qqq++)
			{
				sum_dector += motion2color.data[flow.rows*ROW_SUM + qqq] / 255;
				
			}
			if(flag_str2_int == 2)  sum_dector = sum_dector/1.5;
			if(flag_str2_int == 3)  sum_dector = sum_dector/0.8;
			IplImage temp = motion2color;
			temp_flow = cvCloneImage(&temp);
			////m_flow
			temp_cimg.CopyOf(temp_flow,temp_flow->nChannels);
			temp_cimg.DrawToHDC(temp_hdc,&temp_rect);
            imwrite("flow.bmp", motion2color);  
			//ReleaseDC( temp_pDC );
			sum_dector = sum_dector/flow.rows;
			sum_sum_dection += sum_dector;
			sum_dector = 0;

			// XIANSHI
				CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//根据ID获得窗口指针再获取与该窗口关联的上下文指针
	        HDC hdc= pDC->GetSafeHdc();                      // 获取设备上下文句柄
	        CRect rect;
            // 矩形类
            GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //获取box1客户区
            CvvImage cimg;
            IplImage *src; // 定义IplImage指针变量src     
            src = cvLoadImage("C:\\Users\\icie\\Desktop\\video\\video\\flow.bmp",-1); // 将src指向当前工程文件目录下的图像me.bmp    
            //cimg.CopyOf(src,src->nChannels);

   cimg.DrawToHDC(hdc,&rect);
//输出图像
   ReleaseDC( pDC );
   cimg.Destroy();

      } 
	  if(waitKey(10)>=0)  
            break;  
      std::swap(prevgray, gray); 

    cimg.CopyOf(frame,frame->nChannels);
	//delay_msec(30);
    cimg.DrawToHDC(hdc,&rect);
  float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //读取该帧在视频中的相对位置
  printf("%f\n",ratio);
  if(!frame)break;
  //cvShowImage("IDC_STATIC",frame);   //显示
  

  if ((frames-11) == i)
  {
	  DECTOR_temp_globvar = (int)sum_sum_dection/ROW_SUM_ratio;
	  sum_sum_dection = 0;
	  int jiancerenshu = DECTOR_temp_globvar;
	  int shijirenshu = 0;
	  int errorrenshu = 0;

	if(flag_str2_int == NULL)  	 shijirenshu = 10; 
 	if(flag_str2_int == 0)  	  shijirenshu = 10; 
	if(flag_str2_int == 1)  	  shijirenshu = 10; 
    if(flag_str2_int == 2)  	 shijirenshu = 6; 
	if(flag_str2_int == 3)  	  shijirenshu = 8; 

	if(shijirenshu >jiancerenshu )   
		errorrenshu = shijirenshu - jiancerenshu;
	if(shijirenshu <= jiancerenshu )   
		errorrenshu = jiancerenshu - shijirenshu;
	// jiance 
	  CString jiancerenshu_CString;
	  jiancerenshu_CString.Format(_T("%d"),jiancerenshu);
	  //shijirenshu 
	  CString shijirenshu_CString;
	  shijirenshu_CString.Format(_T("%d"),shijirenshu);

	  CString errorrenshu_CString;
	  errorrenshu_CString.Format(_T("%d"),errorrenshu);


	  m_jiancerenshu.SetWindowText( jiancerenshu_CString ); 
	  m_error.SetWindowText( errorrenshu_CString); 
	if(flag_str2_int == NULL)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
 	if(flag_str2_int == 0)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
	if(flag_str2_int == 1)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
    if(flag_str2_int == 2)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
	if(flag_str2_int == 3)  	  m_shijirenshu.SetWindowText( shijirenshu_CString); 
  }
  char c = cvWaitKey(vfps);
  if(c == 27 )break;
   }

  ReleaseDC( pDC );
 cvReleaseCapture(&capture);
 cvDestroyWindow("example");
}
