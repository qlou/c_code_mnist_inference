/*******************************************************************************
 * @file
 * File:   conv_layer4.c
 * @author Maurice Peemen <m.c.j.peemen@tue.nl>
 * @author Dongrui She <d.she@tue.nl>
 *
 * @copyright  Eindhoven University of Technology - Electronic Systems Group
 * @date       2013
 * @section    DISCLAIMER
 *             THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR
 *             IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 *             WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *             PURPOSE.
 *
 * @section DESCRIPTION
 *
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define width 28
#define height 28
// #include "utils.h"

/************************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : neuron outputs of the feature maps represented as an image
 * Procedure: perform feed forward computation through the feature extraction layers
     *******************************************************************************/


void run_convolution_layer1(unsigned char in_layer[], unsigned char out_layer[],
                            const float bias[], const float weight[]) {
  int k,l,m,n,r;
  static float y[32*(width+2)*(height+2)];

  //init all values with 0
  for (r=0;r<32;r++){
    for(m=0;m<(width+2)*(height+2);m++){
      y[r*(width+2)*(height+2)+m]=0;
    }
  }
  //init values of feature maps at bias value
  for(r=0; r<32; r++){
    for(m=width+2; m<(width+2)*(height+1); m++){
      if((m%(width+2)!=0) && (m%(width+1)!=0))
        y[r*(width+1)*(height+1)+m]=bias[r];
    }
  }  

  //loop over output feature maps
  for(r=0; r<32; r++){
    //convolve weight kernel with input image
    for(n=0; n<width+2; n++){
      for(m=0; m<height+2; m++){//shift input window over image
        //multiply input window with kernel
        if((n%(width+2)!=0) && (n%(width+1)!=0) && m%(height+2)!=0 && m%(height+1)!=0){
          for(l=0; l<5; l++){
            for(k=0; k<5; k++){
              y[r*(width+2)*(height+2)+m*(width+2)+n] += in_layer[(m-1)*height+n-1] * weight[r*32+k*5+l];
            }
          }
        }
      }
    }
  }
  
  //relu activation function
  for(r=0; r<32*(width+2)*(height+2); r++){
    if(y[r]>0)
      out_layer[r] = y[r];
    else
      out_layer[r] = 0;
  }

  //pooling with stride 2
  for(r=0;r<32;r++)
  	for(n=1;n<(width+2)/2;n++)
  		for(m=1;m<(height+2)/2;m++)
  			if((n%(width+2)!=0) && (n%(width+1)!=0) && m%(height+2)!=0 && m%(height+1)!=0){
  				y_out[(n-1)*height+m] = max(max(out_layer[n*height+m],out_layer[n*height+m+1]),max(out_layer[(n+1)*height+m],out_layer[(n+1)*height+m+1]));
  			}
  
}

/********************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : the neuron outputs computed from the input pattern
 * Procedure: perform feed forward computation through the neural network
 ********************************************************************************/
void run_convolution_layer2(unsigned char in_layer[], unsigned char out_layer[],
                            const float bias[], const float weight[]) {
  int k,l,m,n,q,r,qindex;
  static float y[64*(width/2+2)*(height/2+2)];
  //feature maps are sparse connected therefore connection scheme is used
  //const int qq[60]={0,1,2, 1,2,3, 2,3,4, 3,4,5, 0,4,5, 0,1,5,
  //                  0,1,2,3, 1,2,3,4, 2,3,4,5, 0,3,4,5, 0,1,4,5, 0,1,2,5,
  //                  0,1,3,4, 1,2,4,5, 0,2,3,5, 0,1,2,3,4,5};

  //init all values with 0
  for (r=0;r<64;r++){
    for(m=0;m<(width/2+2)*(height/2+2);m++){
      y[r*(width/2+2)*(height/2+2)+m]=0;
    }
  }
  //init values of feature map at bias value
  for(r=0; r<64; r++){
    for(q=0; m<(width/2+2)*(height/2+2); m++){
      if((m%(width/2+2)!=0) && (m%(width/2+1)!=0))
        y[r*(width/2+2)*(height/2+2)+m]=bias[r];
    }
  }

  //loops over output feature maps with 3 input feature maps
  for(q=0; q<32; q++){
    for(r=0; r<64; r++){//connect with all connected 3 input feature maps
      //qindex=qq[r*3+q];//lookup connection address
      //convolve weight kernel with input image
      for(n=0; n<width/2+2; m++){//shift input window over image
        for(m=0; m<height/2+2; n++){
          if((n%(width/2+2)!=0) && (n%(width/2+1)!=0) && m%(height/2+2)!=0 && m%(height/2+1)!=0){
          //multiply input window with kernel
          for(k=0; k<5; k++){
            for(l=0; l<5; l++){
              y[r*（width/2+2）*(height/2+2)+m*(width/2+2)+n] += in_layer[q*(width/2+1)*(height/2+1)+(m-1)*(width/2+2)+n-1]
                * weight[(r*32+q)*5*5+k*5+l];
              }
            }
          }
        }
      }         
    }
  }


  //relu activation function
  for(r=0; r<64*(width/2+2)*(height/2+2); r++){
    if(y[r]>0)
      out_layer[r] = y[r];
    else
      out_layer[r] = 0;
  }

  //pooling with stride 2
  for(r=0;r<32;r++)
  	for(n=1;n<(width/2+2)/2;n++)
  		for(m=1;m<(height/2+2)/2;m++)
  			if((n%(width/2+2)!=0) && (n%(width/2+1)!=0) && m%(height/2+2)!=0 && m%(height/2+1)!=0){
  				y_out[(n-1)*height/4+m] = max(max(out_layer[n*(height/2+2)+m],out_layer[n*(height/2+2)+m+1]),max(out_layer[(n+1)*(height/2+2)+m],out_layer[(n+1)*(height/2+2)+m+1]));
  			}
  
}

/************************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : the neuron outputs computed from the input pattern
 * Procedure: perform feed forward computation through the neural network
 ************************************************************************************/
void run_convolution_layer3(unsigned char in_layer[], unsigned char out_layer[],
                            const float bias[], const float weight[]) {
  int k,l,m,n,q,r;
  static float y[64*7*7];

  //init values of feature maps at bias value
  for(r=0; r<1024; r++)
      y[r]=bias[r];

  for(q=0;r<1024;q++){
  	for(r=0; q<64; r++){//connect with first 8 feature maps
      for(n=0; n<7; n++){//shift input window over image
        for(m=0; m<7; m++){      
        	y[r] += in_layer[q*7*7+n*height+m] * weight[r*7*7*64+q*7*7+n*height+m]; 
          }
        }
      }
    }           
  }
  

  for(r=0; r<80*173*313; r++){//sigmoid activation function
    out_layer[r]=255.999f/(1+expf(-y[r]/256));
  }
}

/************************************************************************************
 * Input   : input image, coefficients bias and weights, vote array for detected signs
 * Output  : voting histogram for the signs
 * Procedure: perform feed forward computation through the neural network layer
              threshold with neuron output to detect signs at pixel positions
************************************************************************************/
int run_convolution_layer4(unsigned char in_layer[], const float bias[],
                            const float weight[], unsigned int detect[]) {
  int m,n,q,r;
  int detections=0;
  int posx, posy;
  float y;
  int set=0;

  float max;

  //convolve weight kernel with input image
  for(m=0; m<173; m++){//shift input window over image
    for(n=0; n<313; n++){
      //init values of feature map at bias value
      y = bias[0];
      for(q=0; q<80; q++){
        y += in_layer[q*173*313+m*313+n] * weight[q];
      }
      // no sigmoid required sigmoid threshold 0.6 => potential should be
      // inverse -ln(0.6^-1 -1)= 0.405 x 256 = 103.799
      //if (y >= 103.799f){ // if sign detected figure out which sign
	  if (y >= 0.0f){ // if sign detected figure out which sign
        max=0;
        for(r=1; r<8; r++){// check other 7 maps for the stronges sign
          y = bias[r];
          for(q=0; q<80; q++){
            y += in_layer[q*173*313+m*313+n] * weight[r*80+q];
          }
          //if (y>=103.799f && y>max){
		  if (y>=0.0f && y>max){
		    max=y;
            posx=n*4;
			posy=m*4;
			detect[detections*4]=posx;
			detect[detections*4+1]=posy;
			detect[detections*4+2]=r;
			detect[detections*4+3]=100.0f/(1+expf(-y/256));
			set=1;
          }
        }
        if (set==1){//this means that a sign is found
          detections=detections+1;
	      set=0;
        }
      }
    }           
  }
  return detections;
}

void annotate_img(unsigned char img[], unsigned int detectarray[], int detections)
{
  int i,x,y,posx,posy; 
  
  for(i=0; i<detections; i++){
    posx=detectarray[i*4];
	posy=detectarray[i*4+1];
    for(x=0; x<32; x++){
	  img[posy*1280+posx+x]=255;
	  img[(posy+31)*1280+posx+x]=255;
	}
    for(y=1; y<31; y++){
      img[(posy+y)*1280+posx]=255;
	  img[(posy+y)*1280+posx+31]=255;
    }	
  }
}

int main(void) {
  int i;
  const int max_speed[8]={0, 30, 50, 60, 70, 80, 90, 100};
  char imagename[32]; 
  static unsigned char in_image[720*1280];//for input image
  //feature map results due to unroling+2 otherwise writes outside array
  static unsigned char net_layer1[6*358*638];
  static unsigned char net_layer2[16*177*317];
  static unsigned char net_layer3[80*173*313];

  static float bias1[6];  //memory for network coefficients
  static float weight1[6*36];
  static float bias2[16];
  static float weight2[(6*3+9*4+6)*36];
  static float bias3[80];
  static float weight3[25*8*80];
  static float bias4[8];
  static float weight4[80*8]; 
  
  static unsigned int detectarray[3*10];
  int detections;

  clock_t starttime, endtime; //vars for measure computation time

  read_bias1("data/bias01.bin", 6, bias1);
  read_weight1("data/weight01.bin", 6*36, weight1);

  read_bias1("data/bias02.bin", 16, bias2);
  read_weight1("data/weight02.bin", 2160, weight2);

  read_bias1("data/bias03.bin", 80, bias3);
  read_weight1("data/weight03.bin", 25*8*80, weight3);

  read_bias1("data/bias04.bin", 8, bias4);
  read_weight1("data/weight04.bin", 80*8, weight4);

  //compute input name
  sprintf(imagename,"data/test%06d.pgm",46);

  //read image from file
  read_image_pgm(in_image, imagename, 1280, 720);

  //start timer
  starttime=clock();
        
  //perform feed forward operation thourgh the network
  run_convolution_layer1(in_image, net_layer1, bias1, weight1);
  run_convolution_layer2(net_layer1, net_layer2, bias2, weight2);
  run_convolution_layer3(net_layer2, net_layer3, bias3, weight3);
  detections=run_convolution_layer4(net_layer3, bias4, weight4, detectarray);      

  //stop timer
  endtime=clock();
  printf("  Elapsed time is %f s\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);
  
  printf("number of detections = %d\n",detections);
  for(i=0; i<detections; i++){
    printf("detection nr %d = %d km/h, box pos= x %d, y %d, confidence = %d\n",i,max_speed[detectarray[i*4+2]], detectarray[i*4],detectarray[i*4+1],detectarray[i*4+3]);
  }
  
  annotate_img(in_image, detectarray, detections);
  
  write_image_pgm(in_image, "output.pgm", 1280, 720);  
  
  return 0;
}
