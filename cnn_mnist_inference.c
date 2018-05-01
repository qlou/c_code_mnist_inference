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

#define LINE_BUFFER_SIZE 100  //Line buffer size for read write 

/*******************************************************************************
 * Input   : char array containing the filename containing all weights, number of weights
 * Output  : array matrix filled with weights for each feature map
 * Procedure: read all weights from file and strore in array
 ******************************************************************************/
double min(double a, double b) {
    return a<b ? a : b;
}

double max(double a, double b) {
    return a>b ? a : b;
}

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


void read_mnist(const char filename[], int length, float vector[])
{
  int i;
  FILE* finput;
  printf("%s\n",filename);
  finput = fopen(filename , "rb" );
  if (finput==NULL) {fputs ("File error",stderr); exit (13);}
  
  fread(vector, sizeof(float), length, finput);
  for(i=0; i<length; i++){
    vector[i]=vector[i];
  }
  fclose(finput);
}


void read_weight1(const char filename[], int size, float matrix[]) {
  FILE* finput;
    
  finput = fopen(filename , "rb" );
  if (finput==NULL) {fputs ("File error",stderr); exit (13);}
  
  fread(matrix, sizeof(float), size, finput);
  fclose(finput);
}


/************************************************************************************
 * Function: void read_bias(char filename[], int length, float vector[])
 * Input   : char array containing the filename and location for reading, number of bias values this
                is the same as the number of output featuremaps, pointer for output
                * Output  : vector filled with bias weights for each feature map
                * Procedure: read bias weights from file normalize to uchar range and strore on correct possition
                ************************************************************************************/
void read_bias1(const char filename[], int length, float vector[]) {
  int i;
  FILE* finput;
  
  finput = fopen(filename , "rb" );
  if (finput==NULL) {fputs ("File error",stderr); exit (13);}
  
  fread(vector, sizeof(float), length, finput);
  for(i=0; i<length; i++){
    vector[i]=256*vector[i];
  }
  fclose(finput);
}


/************************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : neuron outputs of the feature maps represented as an image
 * Procedure: perform feed forward computation through the feature extraction layers
     *******************************************************************************/


void run_convolution_layer1(float in_layer[], float y_out[],
                            const float bias[], const float weight[]) {
  int k,l,m,n,r;
  float y[32*(width+4)*(height+4)];
  float out_layer[32*(width+4)*(height+4)];

  //init all values with 0
  for (r=0;r<32;r++){
    for(m=0;m<(width+4)*(height+4);m++){
      y[r*(width+4)*(height+4)+m]=0;
    }
  }
  //init values of feature maps at bias value
  for(r=0; r<32; r++){
    for(m=0; m<(width+4)*(height+4); m++){
      if((m%(width+4)!=0) && (m%(width+4)!=1) && (m%(width+4)!=30) && (m%(width+4)!=31) && (m>(width+2)*2) && m<(width+2)*30)
        y[r*(width+4)*(height+4)+m]=bias[r];
    }
  }  
  //loop over output feature maps
  // for(r=0; r<32; r++){
    for(r=0;r<1;r++){
    //convolve weight kernel with input image
    for(n=0; n<width+4; n++){
      for(m=0; m<height+4; m++){//shift input window over image
        //multiply input window with kernel
        if((n%(width+4)!=0) && (n%(width+4)!=1) && (n%(width+4)!=30) && (n%(width+4)!=31) 
        	&& (m%(height+4)!=0) && (m%(height+4)!=0) && (m%(height+4)!=30) && (m%(height+4)!=31)){
          for(l=0; l<5; l++){
            for(k=0; k<5; k++){
              y[r*(width+4)*(height+4)+m*(height+4)+n] += in_layer[(m-2)*height+n-2] * weight[(k+l*5)*32+r];
              // if(m>10 && m<18 && n>10 && n<18)
              	// printf("%f,%f\n",in_layer[(m-2)*height+n-2],weight[(k+l*5)*32+r]);
            }
          }
        }
        //printf("%f\n",y[r*(width+4)*(height+4)+m*(height+4)+n]);
      }
    }
  }


  
  //relu activation function
  for(r=0; r<32*(width+4)*(height+4); r++){
    if(y[r]>0)
      out_layer[r] = y[r];
    else
      out_layer[r] = 0;
  }

  // for(int i=1;i<784;i++){
  //	printf("%d,%f\n",i,out_layer[i]);
  //}
  //pooling with stride 2
  int n_new=-1;
  int m_new;
  int test=0;
  for(r=0;r<32;r++)
  // for(r=0;r<1;r++)
  	for(n=2;n<width+2;n=n+2){
  		n_new++;
  		m_new = -1;
  		for(m=2;m<height+2;m=m+2){
  			test++;
  			m_new++;
  			y_out[r*width/2*height/2+n_new*height/2+m_new] = max(max(out_layer[r*(height+4)*(width+4)+n*(height+4)+m]
  				,out_layer[r*(height+4)*(width+4)+n*(height+4)+m+1]),
  					max(out_layer[r*(height+4)*(width+4)+(n+1)*(height+4)+m],
  						out_layer[r*(height+4)*(width+4)+(n+1)*(height+4)+m+1]));
  			// if(n<28 && m<28){
  			//	printf("%d,%d,%d,%d\n",n*(height+4)+m,n*(height+4)+m+1,(n+1)*(height+4)+m,(n+1)*(height+4)+m+1);
  			//	printf("%f,%f,%f,%f\n",out_layer[n*height+m],out_layer[n*height+m+1],out_layer[(n+1)*height+m],out_layer[(n+1)*height+m+1]);
  			// }
  				// printf("%f\n",y_out[n_new*height/2+m_new]);
  		}
  	}

}

/********************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : the neuron outputs computed from the input pattern
 * Procedure: perform feed forward computation through the neural network
 ********************************************************************************/
void run_convolution_layer2(float in_layer[], float y_out[],
                            const float bias[], const float weight[]) {
  int k,l,m,n,q,r,qindex;
  static float y[64*(width/2+4)*(height/2+4)];
  float out_layer[64*(width/2+4)*(height/2+4)];

  for (r=0;r<64;r++){
    for(m=0;m<(width/2+4)*(height/2+4);m++){
      y[r*(width/2+4)*(height/2+4)+m]=0;
    }
  }
  //init values of feature map at bias value
  for(r=0; r<64; r++){
  	for(m=0;m<(width/2+4)*(height/2+4);m++){
    	if((m%(width/2+4)!=0) && (m%(width/2+4)!=1) && (m%(width/2+4)!=30) && (m%(width/2+4)!=31) && (m>(width/2+4)*2) && m<(width/2+4)*30){
        	y[r*(width/2+4)*(height/2+4)+m]=bias[r];
  		}
    }
  }
  //loops over output feature maps with 3 input feature maps
  for(q=0; q<32; q++){
    for(r=0; r<64; r++){//connect with all connected 3 input feature maps
      //qindex=qq[r*3+q];//lookup connection address
      //convolve weight kernel with input image
      for(n=0; n<width/2+4; n++){//shift input window over image
        for(m=0; m<height/2+4; m++){
          if((n%(width/2+4)!=0) && (n%(width/2+4)!=1) && (m%(width/2+4)!=30) && (m%(width/2+4)!=31) 
          	&& (m>(width/2+4)*2) && (m<(width/2+4)*30)){
          	//multiply input window with kernel
          	for(k=0; k<5; k++){
            	for(l=0; l<5; l++){
              		y[r*(width/4+4)*(height/4+4)+m*(width/4+4)+n] += in_layer[q*(width/2+2)*(height/2+1)+(m-1)*(width/2+2)+n-1]
                		* weight[(r*32+q)*64+(k*5+l)*32*64];
              }
            }
          }
        }
      }         
    }
  }

  double sum=0;
  for(r=0; r<64*(width/4)*(height/4); r++){
  	sum = sum+y_out[r];
  }
  printf("%f\n",sum);

  //relu activation function
  for(r=0; r<64*(width/2+4)*(height/2+4); r++){
    if(y[r]>0)
      out_layer[r] = y[r];
    else
      out_layer[r] = 0;
  }

  //pooling with stride 2
  int n_new=-1, m_new=-1;
  for(r=0;r<64;r++)
  	for(n=2;n<width+2;n=n+2){
  		n_new++;
  		m_new = -1;
  		for(m=2;m<height+2;m=m+2){
  			m_new++;
  			y_out[r*width/4*height/4+n_new*height/4+m_new] = max(max(out_layer[r*(height/2+4)*(width/2+4)+n*(height/2+4)+m]
  				,out_layer[r*(height/2+4)*(width/2+4)+n*(height/2+4)+m+1]),
  					max(out_layer[r*(height/2+4)*(width/2+4)+(n+1)*(height/2+4)+m],
  						out_layer[r*(height/2+4)*(width/2+4)+(n+1)*(height/2+4)+m+1]));  		}
  	}

}

/************************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : the neuron outputs computed from the input pattern
 * Procedure: perform feed forward computation through the neural network
 ************************************************************************************/
void run_convolution_layer3(float in_layer[], float out_layer[],
                            const float bias[], const float weight[]) {
  int k,l,m,n,q,r;
  int num = 64*7*7;
  static float y[64*7*7];

  //init values of feature maps at bias value
  for(r=0; r<1024; r++){
      y[r]=bias[r];
  }

  for(q=0;r<1024;q++){
  	for(r=0; q<64; r++){
      for(n=0; n<7; n++){//shift input window over image
        for(m=0; m<7; m++){      
        	y[r] += in_layer[q*7*7+n*height+m] * weight[r*7*7*64+q*7*7+n*height+m]; 
        }
      }
    }           
  }
  

  //relu activation function
  for(r=0; r<1024; r++){
  	if(y[r]>0)
    	out_layer[r]=y[r];
    else
    	out_layer[r]=0;
  }
}

/************************************************************************************
 * Input   : input image, coefficients bias and weights, vote array for detected signs
 * Output  : voting histogram for the signs
 * Procedure: perform feed forward computation through the neural network layer
              threshold with neuron output to detect signs at pixel positions
************************************************************************************/
void run_convolution_layer4(float in_layer[], const float bias[],
                            const float weight[], float probabilities[]) {
  int m,n,q,r,i;
  int posx, posy;
  float y[10];
  int set=0;

  //init values of feature maps at bias value
  for(r=0; r<10; r++)
      y[r]=bias[r];

  for(r=0;r<10;r++){
  	for(q=0; q<1024; q++){
    	y[r] += in_layer[r*1024+q] * weight[r*1024+q]; 
    }           
  }
  for(i=0;i<10;i++){
  	probabilities[i] = y[i];
  }
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
  // const int max_speed[8]={0, 30, 50, 60, 70, 80, 90, 100};
  char imagename[100]; 
  char file_path[200];
  static float in_image[28*28];//for input image
  //feature map results due to unroling+2 otherwise writes outside array
  static float net_layer1[32*14*14];
  static float net_layer2[64*7*7];
  static float net_layer3[1024];
  float probabilities[10];

  static float bias1[32];  //memory for network coefficients
  static float weight1[32*5*5];
  static float weight[32*5*5];
  static float bias2[64];
  static float weight2[64*5*5];
  static float bias3[1024];
  static float weight3[7*7*64*1024];
  static float bias4[10];
  static float weight4[1024*10]; 


  
  // static unsigned int detectarray[3*10];
  // int detections;
  // clock_t starttime, endtime; //vars for measure computation time

  read_bias1("data/bias1.bin", 32, bias1);
  // read_bias("data/bias1.txt", 32, bias1);
  read_weight1("data/weight1.bin", 32*5*5, weight1);
  // read_weight("data/weight1.txt", 32*5*5, weight1);

  read_bias1("data/bias2.bin", 64, bias2);
  // read_bias("data/bias2.txt", 64, bias2);
  read_weight1("data/weight2.bin", 64*5*5, weight2);
  // read_weight("data/weight2.txt", 64*5*5, weight2);

  read_bias1("data/bias3.bin", 1024, bias3);
  // read_bias("data/bias3.txt", 1024, bias3);
  read_weight1("data/weight3.bin", 7*7*64*1024, weight3);
  // read_weight("data/weight3.txt", 7*7*64*1024, weight3);

  read_bias1("data/bias4.bin", 10, bias4);
  // read_bias("data/bias4.txt", 10, bias4);
  read_weight1("data/weight4.bin", 1024*10, weight4);
  // read_weight("data/weight4.txt", 1024*10, weight4);

  //compute input name
  // sprintf(imagename,"MNIST_images/input0.pgm");
  //imagename = "MNIST_images/input0.pgm";
  //read image from file

  sprintf(file_path,"MNIST_images/image2.bin");
  read_mnist(file_path, 784, in_image);
        
  //perform feed forward operation thourgh the network
  run_convolution_layer1(in_image, net_layer1, bias1, weight1);
  // endtime = clock();
  // printf("%f\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);

  
  run_convolution_layer2(net_layer1, net_layer2, bias2, weight2);
  // endtime = clock();
  // printf("%f\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);  
  run_convolution_layer3(net_layer2, net_layer3, bias3, weight3);

  run_convolution_layer4(net_layer3, bias4, weight4, probabilities); 

  int result=0; 
  float max_number=0;
  for(int i=0;i<10;i++){
  	if(probabilities[i]>max_number){
  		result = i;
  		max_number = probabilities[i];
  	}
  	printf("The probabilities for %d is %f\n", i, probabilities[i]);
  }
  printf(" The classification result is %d\n",result);
    
  return 0;
}
