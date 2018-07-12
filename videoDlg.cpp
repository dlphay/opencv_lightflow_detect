
// videoDlg.cpp : ʵ���ļ�
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

int flag_str2_int = 0;  //ѡȡ��Ƶ��־λ

int DECTOR_temp_globvar = 0;
double sum_dector = 0;
double sum_sum_dection = 0;
Mat prevgray, gray, flow, cflow, frame_mat; 
Mat motion2color; 

Mat hist_equalization_BGR_dlphay(Mat input)
{
	Mat output;
	uchar *dataIn = (uchar *)input.ptr<uchar>(0);//input��ͷָ�룬ָ���0�е�0�����أ���Ϊ������
	uchar *dataOut = (uchar *)output.ptr<uchar>(0);
	const int ncols = input.cols;//��ʾ����ͼ���ж�����
	const int nrows = input.rows;//��ʾ����ͼ���ж�����
	int nchannel = input.channels();//ͨ������һ����3
	int pixnum = ncols*nrows;
	int vData[765] = { 0 };//����R+G+B��ʱ��255+255+255������Ϊ765�����ȼ�
	double vRate[765] = { 0 };
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			vData[dataIn[i*ncols*nchannel + j*nchannel + 0]
				+ dataIn[i*ncols*nchannel + j*nchannel + 1]
				+ dataIn[i*ncols*nchannel + j*nchannel + 2]]++;//��Ӧ�����ȼ�ͳ��
		}
	}
	for (int i = 0; i < 764; i++)
	{
		for (int j = 0; j < i; j++)
		{
			vRate[i] += (double)vData[j] / (double)pixnum;//���
		}
	}
	for (int i = 0; i < 764; i++)
	{
		vData[i] = (int)(vRate[i] * 764 + 0.5);//���й�һ������
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
					+ dataIn[i*ncols*nchannel + j*nchannel + 2]);//�ñ任���ֵ��ȥԭֵ�ĵ����ȼ��Ĳ�ֵ����3�����ÿ��ͨ��Ӧ���仯��ֵ
			int b = dataIn[i*ncols*nchannel + j*nchannel + 0] + amplification / 3 + 0.5;
			int g = dataIn[i*ncols*nchannel + j*nchannel + 1] + amplification / 3 + 0.5;
			int r = dataIn[i*ncols*nchannel + j*nchannel + 2] + amplification / 3 + 0.5;
			if (b > 255) b = 255;//����Խλ�ж�
			if (g > 255) g = 255;
			if (r > 255) r = 255;
			if (r < 0) r = 0;//����Խλ�ж�
			if (g < 0) g = 0;
			if (b < 0) b = 0;
			dataOut[i*ncols*nchannel + j*nchannel + 0] = b;
			dataOut[i*ncols*nchannel + j*nchannel + 1] = g;
			dataOut[i*ncols*nchannel + j*nchannel + 2] = r;
		}
	}
	return output;
}
// ��ʱ����
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
	//at�������ص�  
	for (int i = 1; i<src.rows; ++i)
		for (int j = 1; j < src.cols; ++j) {
			if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1)<src.rows && (j + 1)<src.cols) {//��Ե�����д���  
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
			else {//��Ե��ֵ  
				dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
				dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
				dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
			}
		}
}

//��Ÿ�������ֵ  
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
	for (int gap = 9 / 2; gap > 0; gap /= 2)//ϣ������  
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);
	return arr[4];//������ֵ  
}
//��ֵ�˲�������3*3��
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
	_dst.copyTo(dst);//����  
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
		uchar* ptr = input_image.ptr<uchar>(i);  // �����Ϊһ��һ�е�����  ������ptr
		for (int j = 0; j < input_image.cols - 1; j++)
		{
			int x = ptr[j];
			graylevel[x].push_back(0);//����ط�д�Ĳ��ã������ά����ֻ��Ϊ�˼�¼ÿһ���Ҷ�ֵ�����ظ���  
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
	for (i = 0; i<h; i++){	//��ȡֱ��ͼ
		for (j = 0; j<w; j++){
			histgram[Img.at<uchar>(i, j)]++;
		}
	}
	for (i = 0; i<256; i++){ //omiga��Ϊ�ۼ�ֱ��ͼ
		sum = sum + histgram[i];
		omiga[i] = (double)sum / ((double)h*(double)w);
	}
	sum = 0;
	for (i = 0; i<256; i++){  //p�ۼ�ֱ��ͼǶ��任
		sum = (double)sum + (double)histgram[i] * i / ((double)h*(double)w);
		histgram[i] = sum;
	}
	for (i = 0; i<256; i++){
		if (omiga[i] != 0 && (1 - omiga[i]) != 0){ //��ֹ��ĸΪ0
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

// �Ի�������
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

// ʵ��
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


// CvideoDlg �Ի���



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


// CvideoDlg ��Ϣ�������

BOOL CvideoDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// ��������...���˵�����ӵ�ϵͳ�˵��С�

	// IDM_ABOUTBOX ������ϵͳ���Χ�ڡ�
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

	// ���ô˶Ի����ͼ�ꡣ��Ӧ�ó��������ڲ��ǶԻ���ʱ����ܽ��Զ�
	//  ִ�д˲���
	SetIcon(m_hIcon, TRUE);			// ���ô�ͼ��
	SetIcon(m_hIcon, FALSE);		// ����Сͼ��

	// TODO: �ڴ���Ӷ���ĳ�ʼ������

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
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

// �����Ի��������С����ť������Ҫ����Ĵ���
//  �����Ƹ�ͼ�ꡣ����ʹ���ĵ�/��ͼģ�͵� MFC Ӧ�ó���
//  �⽫�ɿ���Զ���ɡ�

void CvideoDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ���ڻ��Ƶ��豸������

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// ʹͼ���ڹ����������о���
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// ����ͼ��
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//���û��϶���С������ʱϵͳ���ô˺���ȡ�ù��
//��ʾ��
HCURSOR CvideoDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}





void CvideoDlg::OnBnClickedButton1()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
		// TODO: �ڴ���ӿؼ�֪ͨ����������
		// TODO: �ڴ���ӿؼ�֪ͨ����������
		// TODO: �ڴ���ӿؼ�֪ͨ����������
	// TODO: �ڴ���ӿؼ�֪ͨ����������
		// TODO: �ڴ���ӿؼ�֪ͨ����������
	 //IplImage *src; // ����IplImageָ�����src     
  //  src = cvLoadImage("D:\\me.bmp",-1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
  //  cvNamedWindow("me",0);//����һ��������Ϊlena����ʾ����    
  //  cvShowImage("me",src);//��lena�����У���ʾsrcָ����ָ���ͼ��    
  //  cvWaitKey(0);//���޵ȴ�����ͼ������ʾ    
  //  cvDestroyWindow("me");//���ٴ���lena    
  //  cvReleaseImage(&src);//�ͷ�IplImageָ��src   
	CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	 HDC hdc= pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	 CRect rect;
// ������
   GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
   CvvImage cimg;
   IplImage *src; // ����IplImageָ�����src     
    src = cvLoadImage("C:\\Users\\icie\\Desktop\\OpticalFlow\\123.bmp",-1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
   cimg.CopyOf(src,src->nChannels);

   cimg.DrawToHDC(hdc,&rect);
//���ͼ��
   ReleaseDC( pDC );
   cimg.Destroy();
//����
}


void CvideoDlg::OnBnClickedButton2()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
		// TODO: �ڴ���ӿؼ�֪ͨ����������
	 //IplImage *src; // ����IplImageָ�����src     
  //  src = cvLoadImage("D:\\me.bmp",-1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
  //  cvNamedWindow("me",0);//����һ��������Ϊlena����ʾ����    
  //  cvShowImage("me",src);//��lena�����У���ʾsrcָ����ָ���ͼ��    
  //  cvWaitKey(0);//���޵ȴ�����ͼ������ʾ    
  //  cvDestroyWindow("me");//���ٴ���lena    
  //  cvReleaseImage(&src);//�ͷ�IplImageָ��src   

	// ���������Ȼ�ȡ�ļ���·���Լ��ļ�

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

	// TODO: �ڴ���ӿؼ�֪ͨ����������
	
	//m_jiancerenshu.SetWindowText( (CString)"ͳ����...");
	//m_error.SetWindowText( (CString)"������..."); 
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

	 // �ļ���·��
	  const char* s1 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6684.mp4";  // 10
	  const char* s2 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6685.mp4";  //  6
	  const char* s3 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6686.mp4";  //  8
	  CvCapture *capture = NULL;
     //int flag_str2_int = 0;
	 flag_str2_int = _ttoi(str2);
	 str2 = AAAAA;
	// m_shijirenshu.SetWindowText(AAAAA); 
	if(flag_str2_int == NULL)  	  capture = cvCreateFileCapture (s1);  //��ȡ��Ƶ
 	if(flag_str2_int == 0)  	  capture = cvCreateFileCapture (s1);  //��ȡ��Ƶ
	if(flag_str2_int == 1)  	  capture = cvCreateFileCapture (s1);  //��ȡ��Ƶ
    if(flag_str2_int == 2)  	  capture = cvCreateFileCapture (s2);  //��ȡ��Ƶ
	if(flag_str2_int == 3)  	  capture = cvCreateFileCapture (s3);  //��ȡ��Ƶ

	CString str_liuliang1 =(CString)"10"; 
	CString str_liuliang2 =(CString)"6"; 
	CString str_liuliang3 =(CString)"8"; 
	if(flag_str2_int == NULL)   m_shijirenshu.SetWindowText(str2);  
	if(flag_str2_int == 0)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 1)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 2)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 3)   m_shijirenshu.SetWindowText(str2); 
	//m_shijirenshu.SetWindowText(AAAAA); 
	CDC *pDC = GetDlgItem(IDC_STATIC)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	 HDC hdc= pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	 CRect rect;
// ������
   GetDlgItem(IDC_STATIC)->GetClientRect(&rect); //��ȡbox1�ͻ���


     if(capture==NULL) {
     printf("NO capture");    //��ȡ���ɹ������ʶ
    //return 1;
   };    
    double fps=cvGetCaptureProperty(capture, CV_CAP_PROP_FPS );   //��ȡ��Ƶ��֡��
 int vfps = 1000 / fps;                                        //����ÿ֡���ŵ�ʱ��
    printf("%5.1f\t%5d\n",fps,vfps);                             
 double frames=cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);//��ȡ��Ƶ���ж���֡
 printf("frames is %f\n",frames);
 //cvNamedWindow("example",CV_WINDOW_AUTOSIZE);                  //���崰��
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
	   frame = cvQueryFrame( capture );                          //ץȡ֡

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

	   // ROI�ü�
	cvRectangle(frame, pt1, pt2, cvScalar(0, 0, 255, 0), 1, 8, 0);



	// frame ÿһ֡
 
	 frame_mat = frame;

	 // 150 250    60 500
	 Mat img_mat_ROI = frame_mat( Range(150, 230) , Range(60, 500));
	 cvtColor(img_mat_ROI, gray, CV_BGR2GRAY);
	 // ȷ����һ֡����

	  if( prevgray.data && aaa++%2 == 0)  
      {  
		   IplImage * temp_flow;
		   CvvImage temp_cimg;
		   	CDC *temp_pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	        HDC temp_hdc= temp_pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
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
				CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	        HDC hdc= pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	        CRect rect;
            // ������
            GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
            CvvImage cimg;
            IplImage *src; // ����IplImageָ�����src     
            src = cvLoadImage("C:\\Users\\icie\\Desktop\\video\\video\\flow.bmp",-1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
            cimg.CopyOf(src,src->nChannels);

   cimg.DrawToHDC(hdc,&rect);
//���ͼ��
   ReleaseDC( pDC );
   cimg.Destroy();

      } 
	  if(waitKey(10)>=0)  
            break;  
      std::swap(prevgray, gray); 

    cimg.CopyOf(frame,frame->nChannels);
	//delay_msec(30);
    cimg.DrawToHDC(hdc,&rect);
  float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //��ȡ��֡����Ƶ�е����λ��
  printf("%f\n",ratio);
  if(!frame)break;
  //cvShowImage("IDC_STATIC",frame);   //��ʾ
  

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
 //   frame = cvQueryFrame( capture );                          //ץȡ֡
 //   cimg.CopyOf(frame,frame->nChannels);
	//delay_msec(30);
 //   cimg.DrawToHDC(hdc,&rect);
 // float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //��ȡ��֡����Ƶ�е����λ��
 // printf("%f\n",ratio);
 // if(!frame)break;
 // //cvShowImage("IDC_STATIC",frame);   //��ʾ
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

	// TODO: �ڴ���ӿؼ�֪ͨ����������
	
	m_jiancerenshu.SetWindowText( (CString)"ͳ����...");
	m_error.SetWindowText( (CString)"������..."); 
	m_x1.SetWindowText( (CString)"60"); 
	m_y1.SetWindowText( (CString)"150"); 
	m_x2.SetWindowText( (CString)"500"); 
	m_y2.SetWindowText( (CString)"250"); 
}


void CvideoDlg::OnBnClickedButton6()
{
		// TODO: �ڴ���ӿؼ�֪ͨ����������
		// TODO: �ڴ���ӿؼ�֪ͨ����������
	// TODO: �ڴ���ӿؼ�֪ͨ����������
		// TODO: �ڴ���ӿؼ�֪ͨ����������
	 //IplImage *src; // ����IplImageָ�����src     
  //  src = cvLoadImage("D:\\me.bmp",-1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
  //  cvNamedWindow("me",0);//����һ��������Ϊlena����ʾ����    
  //  cvShowImage("me",src);//��lena�����У���ʾsrcָ����ָ���ͼ��    
  //  cvWaitKey(0);//���޵ȴ�����ͼ������ʾ    
  //  cvDestroyWindow("me");//���ٴ���lena    
  //  cvReleaseImage(&src);//�ͷ�IplImageָ��src   
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	CString strFolderPath;  
    BROWSEINFO broInfo = {0};  
    TCHAR szDisplayName[1000] = {0};  
  
    broInfo.hwndOwner = this->m_hWnd;  
    broInfo.pidlRoot = NULL;  
    broInfo.pszDisplayName = szDisplayName;  
    broInfo.lpszTitle = _T("��ѡ�񱣴�·��");  
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
       MessageBox(str,_T("��ѡ����ļ�·��"), MB_OK);
}


void CvideoDlg::OnBnClickedButton3()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	// TODO: �ڴ���ӿؼ�֪ͨ����������
		// TODO: �ڴ���ӿؼ�֪ͨ����������
	 //IplImage *src; // ����IplImageָ�����src     
  //  src = cvLoadImage("D:\\me.bmp",-1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
  //  cvNamedWindow("me",0);//����һ��������Ϊlena����ʾ����    
  //  cvShowImage("me",src);//��lena�����У���ʾsrcָ����ָ���ͼ��    
  //  cvWaitKey(0);//���޵ȴ�����ͼ������ʾ    
  //  cvDestroyWindow("me");//���ٴ���lena    
  //  cvReleaseImage(&src);//�ͷ�IplImageָ��src   

	// ���������Ȼ�ȡ�ļ���·���Լ��ļ�

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

	// TODO: �ڴ���ӿؼ�֪ͨ����������
	
	//m_jiancerenshu.SetWindowText( (CString)"ͳ����...");
	//m_error.SetWindowText( (CString)"������..."); 
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

	 // �ļ���·��
	  const char* s1 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6684.mp4";  // 10
	  const char* s2 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6685.mp4";  //  6
	  const char* s3 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6686.mp4";  //  8
	  CvCapture *capture = NULL;
     //int flag_str2_int = 0;
	 flag_str2_int = _ttoi(str2);
	 str2 = AAAAA;
	// m_shijirenshu.SetWindowText(AAAAA); 
	if(flag_str2_int == NULL)  	  capture = cvCreateFileCapture (s1);  //��ȡ��Ƶ
 	if(flag_str2_int == 0)  	  capture = cvCreateFileCapture (s1);  //��ȡ��Ƶ
	if(flag_str2_int == 1)  	  capture = cvCreateFileCapture (s1);  //��ȡ��Ƶ
    if(flag_str2_int == 2)  	  capture = cvCreateFileCapture (s2);  //��ȡ��Ƶ
	if(flag_str2_int == 3)  	  capture = cvCreateFileCapture (s3);  //��ȡ��Ƶ

	CString str_liuliang1 =(CString)"10"; 
	CString str_liuliang2 =(CString)"6"; 
	CString str_liuliang3 =(CString)"8"; 
	if(flag_str2_int == NULL)   m_shijirenshu.SetWindowText(str2);  
	if(flag_str2_int == 0)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 1)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 2)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 3)   m_shijirenshu.SetWindowText(str2); 
	//m_shijirenshu.SetWindowText(AAAAA); 
	CDC *pDC = GetDlgItem(IDC_STATIC)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	 HDC hdc= pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	 CRect rect;
// ������
   GetDlgItem(IDC_STATIC)->GetClientRect(&rect); //��ȡbox1�ͻ���


     if(capture==NULL) {
     printf("NO capture");    //��ȡ���ɹ������ʶ
    //return 1;
   };    
    double fps=cvGetCaptureProperty(capture, CV_CAP_PROP_FPS );   //��ȡ��Ƶ��֡��
 int vfps = 1000 / fps;                                        //����ÿ֡���ŵ�ʱ��
    printf("%5.1f\t%5d\n",fps,vfps);                             
 double frames=cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);//��ȡ��Ƶ���ж���֡
 printf("frames is %f\n",frames);
 //cvNamedWindow("example",CV_WINDOW_AUTOSIZE);                  //���崰��
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
	   frame = cvQueryFrame( capture );                          //ץȡ֡

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

	   // ROI�ü�
	cvRectangle(frame, pt1, pt2, cvScalar(0, 0, 255, 0), 1, 8, 0);



	// frame ÿһ֡
 
	 frame_mat = frame;

	 // 150 250    60 500
	 Mat img_mat_ROI = frame_mat( Range(150, 230) , Range(60, 500));
	 cvtColor(img_mat_ROI, gray, CV_BGR2GRAY);
	 // ȷ����һ֡����

	  if( prevgray.data && aaa++%2 == 0)  
      {  
		   IplImage * temp_flow;
		   CvvImage temp_cimg;
		   	CDC *temp_pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	        HDC temp_hdc= temp_pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
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
				CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	        HDC hdc= pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	        CRect rect;
            // ������
            GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
            CvvImage cimg;
            IplImage *src; // ����IplImageָ�����src     
            src = cvLoadImage("C:\\Users\\icie\\Desktop\\video\\video\\flow.bmp",-1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
            cimg.CopyOf(src,src->nChannels);

   cimg.DrawToHDC(hdc,&rect);
//���ͼ��
   ReleaseDC( pDC );
   cimg.Destroy();

      } 
	  if(waitKey(10)>=0)  
            break;  
      std::swap(prevgray, gray); 

    cimg.CopyOf(frame,frame->nChannels);
	//delay_msec(30);
    cimg.DrawToHDC(hdc,&rect);
  float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //��ȡ��֡����Ƶ�е����λ��
  printf("%f\n",ratio);
  if(!frame)break;
  //cvShowImage("IDC_STATIC",frame);   //��ʾ
  

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
 //   frame = cvQueryFrame( capture );                          //ץȡ֡
 //   cimg.CopyOf(frame,frame->nChannels);
	//delay_msec(30);
 //   cimg.DrawToHDC(hdc,&rect);
 // float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //��ȡ��֡����Ƶ�е����λ��
 // printf("%f\n",ratio);
 // if(!frame)break;
 // //cvShowImage("IDC_STATIC",frame);   //��ʾ
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
	// TODO: �ڴ���ӿؼ�֪ͨ����������
		// TODO: �ڴ���ӿؼ�֪ͨ����������
	// TODO: �ڴ���ӿؼ�֪ͨ����������
		// TODO: �ڴ���ӿؼ�֪ͨ����������
	 //IplImage *src; // ����IplImageָ�����src     
  //  src = cvLoadImage("D:\\me.bmp",-1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
  //  cvNamedWindow("me",0);//����һ��������Ϊlena����ʾ����    
  //  cvShowImage("me",src);//��lena�����У���ʾsrcָ����ָ���ͼ��    
  //  cvWaitKey(0);//���޵ȴ�����ͼ������ʾ    
  //  cvDestroyWindow("me");//���ٴ���lena    
  //  cvReleaseImage(&src);//�ͷ�IplImageָ��src   

	// ���������Ȼ�ȡ�ļ���·���Լ��ļ�

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

	// TODO: �ڴ���ӿؼ�֪ͨ����������
	
	//m_jiancerenshu.SetWindowText( (CString)"ͳ����...");
	//m_error.SetWindowText( (CString)"������..."); 
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

	 // �ļ���·��
	  const char* s1 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6684.mp4";  // 10
	  const char* s2 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6685.mp4";  //  6
	  const char* s3 = "C:\\Users\\icie\\Desktop\\video\\video_mp4\\IMG_6686.mp4";  //  8
	  CvCapture *capture = NULL;
     //int flag_str2_int = 0;
	 flag_str2_int = _ttoi(str2);
	 str2 = AAAAA;
	// m_shijirenshu.SetWindowText(AAAAA); 
	if(flag_str2_int == NULL)  	  capture = cvCreateFileCapture (s1);  //��ȡ��Ƶ
 	if(flag_str2_int == 0)  	  capture = cvCreateFileCapture (s1);  //��ȡ��Ƶ
	if(flag_str2_int == 1)  	  capture = cvCreateFileCapture (s1);  //��ȡ��Ƶ
    if(flag_str2_int == 2)  	  capture = cvCreateFileCapture (s2);  //��ȡ��Ƶ
	if(flag_str2_int == 3)  	  capture = cvCreateFileCapture (s3);  //��ȡ��Ƶ

	CString str_liuliang1 =(CString)"10"; 
	CString str_liuliang2 =(CString)"6"; 
	CString str_liuliang3 =(CString)"8"; 
	if(flag_str2_int == NULL)   m_shijirenshu.SetWindowText(str2);  
	if(flag_str2_int == 0)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 1)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 2)   m_shijirenshu.SetWindowText(str2); 
	if(flag_str2_int == 3)   m_shijirenshu.SetWindowText(str2); 
	//m_shijirenshu.SetWindowText(AAAAA); 
	CDC *pDC = GetDlgItem(IDC_STATIC)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	 HDC hdc= pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	 CRect rect;
// ������
   GetDlgItem(IDC_STATIC)->GetClientRect(&rect); //��ȡbox1�ͻ���


     if(capture==NULL) {
     printf("NO capture");    //��ȡ���ɹ������ʶ
    //return 1;
   };    
    double fps=cvGetCaptureProperty(capture, CV_CAP_PROP_FPS );   //��ȡ��Ƶ��֡��
 int vfps = 1000 / fps;                                        //����ÿ֡���ŵ�ʱ��
    printf("%5.1f\t%5d\n",fps,vfps);                             
 double frames=cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);//��ȡ��Ƶ���ж���֡
 printf("frames is %f\n",frames);
 //cvNamedWindow("example",CV_WINDOW_AUTOSIZE);                  //���崰��
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
	   frame = cvQueryFrame( capture );                          //ץȡ֡

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

	   // ROI�ü�
	cvRectangle(frame, pt1, pt2, cvScalar(0, 0, 255, 0), 1, 8, 0);



	// frame ÿһ֡
 
	 frame_mat = frame;

	 // 150 250    60 500
	 Mat img_mat_ROI = frame_mat( Range(150, 230) , Range(60, 500));
	 cvtColor(img_mat_ROI, gray, CV_BGR2GRAY);
	 // ȷ����һ֡����

	  if( prevgray.data && aaa++%2 == 0)  
      {  
		   IplImage * temp_flow;
		   CvvImage temp_cimg;
		   	CDC *temp_pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	        HDC temp_hdc= temp_pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
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
				CDC *pDC = GetDlgItem(IDC_STATIC_FLOW)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ��
	        HDC hdc= pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��
	        CRect rect;
            // ������
            GetDlgItem(IDC_STATIC_FLOW)->GetClientRect(&rect); //��ȡbox1�ͻ���
            CvvImage cimg;
            IplImage *src; // ����IplImageָ�����src     
            src = cvLoadImage("C:\\Users\\icie\\Desktop\\video\\video\\flow.bmp",-1); // ��srcָ��ǰ�����ļ�Ŀ¼�µ�ͼ��me.bmp    
            //cimg.CopyOf(src,src->nChannels);

   cimg.DrawToHDC(hdc,&rect);
//���ͼ��
   ReleaseDC( pDC );
   cimg.Destroy();

      } 
	  if(waitKey(10)>=0)  
            break;  
      std::swap(prevgray, gray); 

    cimg.CopyOf(frame,frame->nChannels);
	//delay_msec(30);
    cimg.DrawToHDC(hdc,&rect);
  float ratio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);     //��ȡ��֡����Ƶ�е����λ��
  printf("%f\n",ratio);
  if(!frame)break;
  //cvShowImage("IDC_STATIC",frame);   //��ʾ
  

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
