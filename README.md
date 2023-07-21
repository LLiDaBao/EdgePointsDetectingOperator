# EdgePointsDetectingOperator
功能: 抓边算子。在指定ROI(矩形或环形区域)内, 沿主轴(轮廓线)的边缘点提取, 以及检测椭圆边缘。    
extract pre-defined region (rectangle or annulus) edge points along the profilie line and detect 


___亚像素边缘点提取___  
[1]: 预定义一个ROI区域;  
[2]: 沿垂直轮廓线方向, 以一定步长计算平均灰度值;  
[3]: 内插计算平均灰度的一阶导数并进行缩放;  
[4]: 过滤并提取边缘关键点, 执行Devernay方法亚像素化;   

___1D Sub-Pixel Edge Points Extraction___    
[1]: Generate Rotated Rectangle or Annulus   
[2]: Calculate Avarage Gray Value along the Direction Perpendicular to Profile Line  
[3]: Gaussian Blur and Calculate 1st Derivation  
[4]: Threshold and Select by Devernay method   
  
___椭圆检测___  
[1]: 创建映射将图片沿极坐标展平;  
[2]: 动态规划在展平图片中寻找局部极值点;  
[3]: 逆映射这些点为原坐标系中, 结果即为椭圆边缘点, 拟合;

___Ellipse Detect___   
[1]: Map Source Image Pixels to New Postion like Flatten Image  
[2]: Dynamic Programming to Search Local Maximum Postion  
[3]: Remap Those Points to Original Postioin. These Points are Ellipse Edge, You Could Fit Them if U Like  
__Result__  
*test1*:  
![image](https://github.com/LLiDaBao/EdgePointsDetectingOperator/blob/master/images/test.jpg)  

*result1*:  
![image](https://github.com/LLiDaBao/EdgePointsDetectingOperator/blob/master/images/result.jpg)  

*test2*:  
![image](https://github.com/LLiDaBao/EdgePointsDetectingOperator/blob/master/images/test2.png)  

*result2*:  
![image](https://github.com/LLiDaBao/EdgePointsDetectingOperator/blob/master/images/result2.jpg)
