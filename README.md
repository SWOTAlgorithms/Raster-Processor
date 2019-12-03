 Description of SWOT Rasterization Software

1. Introduction
This document describes the conversion of variables from SWOT pixel-cloud data produced by the SWOT hydrology simulator to the SWOT raster product. The variables included in the raster product are: water fraction, water fraction uncertainty, height, height uncertainty and cross track distance in current version. Other variables such as frozen flag, geoid height and dry/wet tropo correction, which need input from future iterations of the pixel cloud product, are set as null values. This document focuses on the input, output and equations used in the software for rasterization.

2. How to setup and run the software
The software consists of two files (aggregate.py and to_raster.py). All parameters for running the software are configured in  to_raster.py. In general, the parameters required to run the code are the path of input pixel cloud files and the desired spatial resolution. For running the code, just simply type ‘python to_raster.py’ in the command prompt.

![alt text](https://github.com/tamuzhang/Raster-Processor/blob/master/img/Fig1.png)

Fig. 1. Flowchart


3. Aggregation of pixel cloud points to raster grid
The first step to convert pixel-cloud product to raster is re-projection (Fig. 1). Pixels under GEO lat/lon are re-projected to the appropriate UTM projection. Since each raster grid may contain multiple pixel-cloud pixels, an aggregation operation is needed. Details of inputs and outputs of each variable of aggregation are provided in appendix.

3.1 Height and water area
The equation for calculating water fraction in raster product is in equation (1), where hi is the height of the ith grid in raster product. h(x) is the height for pixel-cloud pixel x, which is assigned to the ith grid in raster product. N is total number of pixel-cloud pixels which are assigned to the ith grid of raster product (N can be calculated through re-projection).

h_i=  (∑_x h(x))/N
	(1)
The aggregations of water area are different for interior water pixels and edge pixels (Williams 2018). The water areas of interior water pixels are aggregated directly over a raster grid. The areas of water pixels near land and land pixels near water are calculated using a water-fraction based approach. The calculation can be expressed as equation (2),

A_i=∑_x A(x)(I_(dw,in) (x)+α(x)I_de (x))
	(2)

where Ai is the water area of the ith raster grid. Idw,in(x) stands for interior water pixels from pixel-cloud products, and Ide(x) indicates edge pixels of pixel-cloud products. A(x) is the area of pixel-cloud pixel, α(x) is the water fraction of edge pixel in pixel-cloud product. An example of water fraction over Sacramento river in Fig. 2 shows the water fractions of the river centerline are usually higher than pixels along river edge. Note that water faction of some pixels may exceed 1 due to noise in the SWOT pixel cloud inundation extent calculation. At the current stage, we retain these values in the raster product so that the summation across many raster cells will remain unbiased.
![alt text](https://github.com/tamuzhang/Raster-Processor/blob/master/img/Fig2.png)

Fig. 2 Water fraction on SWOT raster image over Sacramento river

3.2 Uncertainty estimates
The uncertainty of water fraction and water height are quantified in the raster product by using the algorithm proposed by Williams (2018). The input variable needed for estimating the height and water area uncertainties (e. g. probability of detecting water when there is no water, missed detection rate, correct detection rate) are provided in the pixel-cloud product.

3.3 Cross-track distances
The aggregation of cross crack distances is similar with water height in 3.1. A simple average is chosen to aggregate the cross-track distances from pixel-cloud pixels to raster grids.

Appendix  
Output in raster product	Input from pixel-cloud product
height 	height, pixel classification
water fraction	pixel area, water fraction, pixel classification
height uncertainty	height, pixel classification, two channel powers, number of rare looks, scale factor to get effective looks, height sensitivity to phase
water fraction uncertainty	pixel area, water fraction, water fraction uncertainty, pixel classification, probability of false detection of water, probability of missed detection of water, probability of correct assignment, probability of water (prior)
cross-track distance	cross track, pixel classification

References

B. Williams, “SWOT Hydrology Height and Area Uncertainty Estimation,” Jet Propulsion Lab, Tech. Rep., 2018.
