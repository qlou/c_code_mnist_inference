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
    return a<b ? a : b;
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

void read_image(float image_float[], char filename[], int imageWidth, int imageHeight)
{   /************************************************************************************
     * Function: void read_image_pgm(unsigned char image[], char filename[], int imageWidth, int imageHeight)
     * Input   : uchar array pointer for output result, char array with filename, int with with, int with height
     * Output  : uchar image array
     * Procedure: if image dimensions and layout pgm correct imare is read from file to image array
     ************************************************************************************/
  int grayMax;
  int PGM_HEADER_LINES=3;
  FILE* input;
  unsigned char image[30*30];

  int headerLines = 1;
  int scannedLines= 0;
  long int counter =0;

  //read header strings
  char *lineBuffer = (char *) malloc(LINE_BUFFER_SIZE+1);
  char *split;
  char *format = (char *) malloc(LINE_BUFFER_SIZE+1);
  char P5[]="P5";
  char comments[LINE_BUFFER_SIZE+1];

  //open the input PGM file
  input=fopen(filename, "rb");

  //read the input PGM file header
  while(scannedLines < headerLines){
    fgets(lineBuffer, LINE_BUFFER_SIZE, input);
    //if not comments
    if(lineBuffer[0] != '#'){
      scannedLines += 1;
      //read the format
      if(scannedLines==1){
        split=strtok(lineBuffer, " \n");
        strcpy(format,split);
        if(strcmp(format,P5) == 0){
          //printf("FORMAT: %s\n",format);
          headerLines=PGM_HEADER_LINES;
        }
        else
        {
          printf("Only PGM P5 format is support. \n");
        }
      }
      //read width and height
      if (scannedLines==2)
      {
        split=strtok(lineBuffer, " \n");
        if(imageWidth == atoi(split)){ //check if width matches description
          //printf("WIDTH: %d, ", imageWidth);
        }
        else{
          printf("input frame has wrong width should be WIDTH: %d, ", imageWidth);
          exit(4);
        }
        split = strtok (NULL, " \n");
        if(imageHeight == atoi(split)){ //check if heigth matches description
          //printf("HEIGHT: %d\n", imageHeight);
        }
        else{
          printf("input frame has wrong height should be HEIGHT: %d, ", imageHeight);
          exit(4);
        }
      }
      // read maximum gray value
      if (scannedLines==3)
      {
        split=strtok(lineBuffer, " \n");
        grayMax = atoi(split);
        //printf("GRAYMAX: %d\n", grayMax);
      }
    }
    else
    {
      strcpy(comments,lineBuffer);
      //printf("comments: %s", comments);
    }
  }

  counter = fread(image, sizeof(unsigned char), imageWidth * imageHeight, input);
  //printf("pixels read: %d\n",counter);
  int i;
  for (i=0;i<784;i++){
  	image_float[i] = (float)(image[i])/255;   
  }
  
  //for(i=0;i<784;i++){
  //	printf("%f,",image_float[i]);
  //}
  //close the input pgm file and free line buffer
  fclose(input);
  free(lineBuffer);
  free(format);
}

/************************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : neuron outputs of the feature maps represented as an image
 * Procedure: perform feed forward computation through the feature extraction layers
     *******************************************************************************/


void run_convolution_layer1(float in_layer[], float y_out[],
                            const float bias[], const float weight[]) {
  int k,l,m,n,r;
  static float y[32*(width+2)*(height+2)];
  float out_layer[32*(width+2)*(height+2)];

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
void run_convolution_layer2(float in_layer[], float y_out[],
                            const float bias[], const float weight[]) {
  int k,l,m,n,q,r,qindex;
  static float y[64*(width/2+2)*(height/2+2)];
  float out_layer[64*(width/2+2)*(height/2+2)];
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
    for(m=0; m<(width/2+2)*(height/2+2); m++){
      if((m%(width/2+2)!=0) && (m%(width/2+1)!=0))
        y[r*(width/2+2)*(height/2+2)+m]=bias[r];
    }
  }
  printf("sss\n");
  //loops over output feature maps with 3 input feature maps
  for(q=0; q<32; q++){
    for(r=0; r<64; r++){//connect with all connected 3 input feature maps
      //qindex=qq[r*3+q];//lookup connection address
      //convolve weight kernel with input image
      for(n=0; n<width/2+2; n++){//shift input window over image
        for(m=0; m<height/2+2; m++){
          if((n%(width/2+2)!=0) && (n%(width/2+1)!=0) && m%(height/2+2)!=0 && m%(height/2+1)!=0){
          //multiply input window with kernel
          for(k=0; k<5; k++){
            for(l=0; l<5; l++){
            	//printf("ss\n");
              y[r*(width/2+2)*(height/2+2)+m*(width/2+2)+n] += in_layer[q*(width/2+1)*(height/2+1)+(m-1)*(width/2+2)+n-1]
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
  static float in_image[28*28];//for input image
  //feature map results due to unroling+2 otherwise writes outside array
  static float net_layer1[32*14*14];
  static float net_layer2[64*7*7];
  static float net_layer3[1024];
  float probabilities[10];

  static float bias1[32];  //memory for network coefficients
  static float weight1[32*5*5];
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
  read_weight1("data/weight1.bin", 32*5*5, weight1);

  read_bias1("data/bias2.bin", 64, bias2);
  read_weight1("data/weight2.bin", 64*5*5, weight2);

  read_bias1("data/bias3.bin", 1024, bias3);
  read_weight1("data/weight3.bin", 7*7*64*1024, weight3);

  read_bias1("data/bias4.bin", 10, bias4);
  read_weight1("data/weight4.bin", 1024*10, weight4);

  //compute input name
  sprintf(imagename,"MNIST_images/input0.pgm");
  //imagename = "MNIST_images/input0.pgm";
  //read image from file
  read_image(in_image, imagename, 28, 28);

  for(i=0;i<784;i++){
  	printf("%f,",weight1[i]);
  }

  // endtime = clock();
  // printf("%f\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);

  // start timer
  // starttime=clock();
        
  //perform feed forward operation thourgh the network
  run_convolution_layer1(in_image, net_layer1, bias1, weight1);

  // endtime = clock();
  // printf("%f\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);
  printf("\n");

  
  run_convolution_layer2(net_layer1, net_layer2, bias2, weight2);
  // endtime = clock();
  // printf("%f\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);  
  run_convolution_layer3(net_layer2, net_layer3, bias3, weight3);

  run_convolution_layer4(net_layer3, bias4, weight4, probabilities); 



  //stop timer
  //endtime=clock();

  int result=0, max=0;
  for(int i=0;i<10;i++){
  	if(probabilities[i]>max){
  		result = i;
  		max = probabilities[i];
  	}
  	printf("The probabilities for %d is %f\n", i,probabilities[i]);
  }
  printf(" The classification result is %d\n",result);
    
  return 0;
}
