
// videoDlg.h : ͷ�ļ�
//

#pragma once
#include "afxwin.h"


// CvideoDlg �Ի���
class CvideoDlg : public CDialogEx
{
// ����
public:
	CvideoDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
	enum { IDD = IDD_VIDEO_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��


// ʵ��
protected:
	HICON m_hIcon;

	// ���ɵ���Ϣӳ�亯��
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton5();
	afx_msg void OnBnClickedButton6();
	CEdit m_folderDir;
	CEdit m_edit5;
	CEdit m_x1;
	CEdit m_y1;
	CEdit m_x2;
	CEdit m_y2;
	CEdit m_shijirenshu;
	CEdit m_jiancerenshu;
	CEdit m_error;
	CStatic m_flow;
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton4();
};
