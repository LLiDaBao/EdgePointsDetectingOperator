# EdgePointsDetectingOperator
功能: 在指定ROI(矩形或环形区域)内, 沿主轴(轮廓线)的边缘点提取  
extract pre-defined region (rectangle or annulus) edge points along the profilie line  


___工作流程___  
[1]: 预定义一个ROI区域;  
[2]: 沿垂直轮廓线方向, 以一定步长计算平均灰度值;  
[3]: 内插计算平均灰度的一阶导数并进行缩放;  
[4]: 过滤并提取边缘关键点;   

___1D Edge Points Extraction___    
[1]: Generate Rotated Rectangle or Annulus   
[2]: Calculate Avarage Gray Value along the Direction Perpendicular to Profile Line  
[3]: Gaussian Blur and Calculate 1st Derivation  
[4]: Threshold and Select   
	
__Result__  
//待添加
