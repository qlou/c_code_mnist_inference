
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define width 28
#define height 28

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


void read_mnist(const char filename[], int length, float matrix[])
{
  int i;
 
  FILE* finput;
  
  finput = fopen(filename , "rb" );
  
  if (finput==NULL) {fputs ("File error",stderr); exit (13);}

  fread(matrix, sizeof(float), length, finput);

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

  
  float in_layer_padding[32*32*32];
  // add padding to input
  for (r=0;r<32;r++){
    for(m=0; m<height+4; m++){
      for(n=0; n<width+4; n++){
        if((n%(width+4)!=0) && (n%(width+4)!=1) && (n%(width+4)!=30) && (n%(width+4)!=31) 
        	&& (m%(height+4)!=0) && (m%(height+4)!=1) && (m%(height+4)!=30) && (m%(height+4)!=31)){
                 in_layer_padding[r*(width+4)*(height+4)+m*(width+4)+n] = in_layer[(m-2)*width+n-2];
        }
        else
             in_layer_padding[r*(width+4)*(height+4)+m*(width+4)+n] = 0;
      }
    }
  }



  //loop over output feature maps
  for(r=0;r<32;r++){
    //convolve weight kernel with input image
    for(m=0; m<height+4; m++){
      for(n=0; n<width+4; n++){//shift input window over image
        //multiply input window with kernel
        if((n%(width+4)!=0) && (n%(width+4)!=1) && (n%(width+4)!=30) && (n%(width+4)!=31) 
        	&& (m%(height+4)!=0) && (m%(height+4)!=1) && (m%(height+4)!=30) && (m%(height+4)!=31)){
          for(l=0; l<5; l++){
            for(k=0; k<5; k++){
              y[r*(width+4)*(height+4)+m*(width+4)+n] += in_layer_padding[(m+l-2)*(width+4)+n+k-2] * weight[(k+l*5)*32+r];
            }
          }
        
        }
      }
    }
  }


  //init values of feature maps at bias value
  for(r=0; r<32; r++){
    for(m=0; m<height+4; m++){
      for(n=0; n<width+4; n++){
         if((n%(width+4)!=0) && (n%(width+4)!=1) && (n%(width+4)!=30) && (n%(width+4)!=31) 
        	&& (m%(height+4)!=0) && (m%(height+4)!=1) && (m%(height+4)!=30) && (m%(height+4)!=31)){
              y[r*(width+4)*(height+4)+m*(width+4)+n] += bias[r];
         }
      }
    }
  }  

  //relu activation function
  for(r=0; r<32;r++)
    for(m=0;m<(width+4)*(height+4);m++){
    if(y[r*(width+4)*(height+4)+m]>0)
      out_layer[r*(width+4)*(height+4)+m] = y[r*(width+4)*(height+4)+m];
    else
      out_layer[r*(width+4)*(height+4)+m] = 0;
  }

  int n_new=-1;
  int m_new;
  int test=0;
  for(r=0;r<32;r++){
  //for(r=0;r<1;r++){
    m_new = -1;
  	for(m=2;m<height+2;m=m+2){
  	//for(m=2;m<3;m=m+2){
     	m_new++;
  		n_new = -1;
  		for(n=2;n<width+2;n=n+2){
        //for(n=2;n<width+2;n=n+2){
  			test++;
  			n_new++;
            // printf("%d,%d,%d,%d,%d,%d\n",n,m,r,n_new,m_new,r*width/2*height/2+n_new*height/2+m_new);
  			y_out[r*width/2*height/2+m_new*width/2+n_new] = max(max(out_layer[r*(height+4)*(width+4)+m*(width+4)+n]
  				,out_layer[r*(height+4)*(width+4)+m*(width+4)+n+1]),
  					max(out_layer[r*(height+4)*(width+4)+(m+1)*(width+4)+n],
  						out_layer[r*(height+4)*(width+4)+(m+1)*(width+4)+n+1]));
  		}
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


  float in_layer_padding[32*18*18];
  // add padding to input
   for (r=0;r<32;r++){
  // for(r=0;r<1;r++){
    for(m=0; m<height/2+4; m++){
      for(n=0; n<width/2+4; n++){
        if((n%(width/2+4)!=0) && (n%(width/2+4)!=1) && (n%(width/2+4)!=(width/2+2)) && (n%(width/2+4)!=(width/2+3)) 
        	&& (m%(height/2+4)!=0) && (m%(height/2+4)!=1) && (m%(height/2+4)!=width/2+2) && (m%(height/2+4)!=width/2+3)){
                 in_layer_padding[r*(width/2+4)*(height/2+4)+m*(width/2+4)+n] = in_layer[r*(height/2)*(width/2)+(m-2)*width/2+n-2];
        }
        else
             in_layer_padding[r*(width/2+4)*(height/2+4)+m*(width/2+4)+n] = 0;
      }
    }
  }


  //for(r=0; r<64; r++){
  for(r=0;r<64; r++){
   for(q=0;q<32;q++){ 
      for(m=0; m<height/2+4; m++){//shift input window over image
      //for(m=0; m<3; m++){
        for(n=0; n<width/2+4; n++){
        //for(n=0; n<3; n++){
          if((n%(width/2+4)!=0) && (n%(width/2+4)!=1) && (n%(width/2+4)!=16) && (n%(width/2+4)!=17) 
          	&& (m%(width/2+4)!=0) && (m%(width/2+4)!=1) && (m%(width/2+4)!=16) && (m%(width/2+4)!=17)) {
          	//multiply input window with kernel
            float fm=0;
          	for(k=0; k<5; k++){
            	for(l=0; l<5; l++){
              		y[r*(width/2+4)*(height/2+4)+m*(width/2+4)+n] += in_layer_padding[q*(height/2+4)*(width/2+4)+(m+l-2)*(width/2+4)+n+k-2]
                		* weight[q*64+r+64*32*(k+l*5)];
              }
            }
          }
        }
      }         
    }
  }

  //init values of feature maps at bias value
  for(r=0; r<64; r++){
    for(m=0; m<height/2+4; m++){
      for(n=0; n<width/2+4; n++){
         if((n%(width/2+4)!=0) && (n%(width/2+4)!=1) && (n%(width/2+4)!=16) && (n%(width/2+4)!=17) 
        	&& (m%(height/2+4)!=0) && (m%(height/2+4)!=1) && (m%(height/2+4)!=16) && (m%(height/2+4)!=17)){
              y[r*(width/2+4)*(height/2+4)+m*(width/2+4)+n] += bias[r];
         }
      }
    }
  }  

  //relu activation function
  for(r=0; r<64;r++){
      for(m=0;m<(width/2+4)*(height/2+4); m++){
         if(y[r*(width/2+4)*(height/2+4)+m]>0)
            out_layer[r*(width/2+4)*(height/2+4)+m] = y[r*(width/2+4)*(height/2+4)+m];
         else
            out_layer[r*(width/2+4)*(height/2+4)+m] = 0;
     }
  }



  //pooling with stride 2
  int n_new=-1, m_new=-1;
  for(r=0;r<64;r++){
    m_new = -1;
  	for(m=2;m<height+2;m=m+2){
     	m_new++;
  		n_new = -1;
  		for(n=2;n<width+2;n=n+2){
  			n_new++;
  			y_out[r*width/4*height/4+m_new*width/4+n_new] = max(max(out_layer[r*(height/2+4)*(width/2+4)+m*(width/2+4)+n]
  				,out_layer[r*(height/2+4)*(width/2+4)+m*(width/2+4)+n+1]),
  					max(out_layer[r*(height/2+4)*(width/2+4)+(m+1)*(width/2+4)+n],
  						out_layer[r*(height/2+4)*(width/2+4)+(m+1)*(width/2+4)+n+1]));
        }
    }
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

  for(r=0;r<1024;r++){
  	for(q=0; q<64; q++){
      for(m=0; m<height/4; m++){//shift input window over image
        for(n=0; n<height/4; n++){
        	y[r] += in_layer[q*(height/4)*(width/4)+m*width/4+n] * weight[r+q*1024+(m*(height/4)+n)*64*1024]; 
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
    	y[r] += in_layer[q] * weight[r+q*10]; 
    }           
  }
  for(i=0;i<10;i++){
  	probabilities[i] = y[i];
  }

}

int main(int argc, char *argv[]) {


  int i;

  char imagename[100]; 
  char file_path[200];
  static float in_image[28*28];//for input image

  static float net_layer1[32*14*14];
  static float net_layer2[64*7*7];
  static float net_layer3[1024];
  float probabilities[10];

  static float bias1[32];  //memory for network coefficients
  static float weight1[32*5*5];
  static float weight[32*5*5];
  static float bias2[64];
  static float weight2[32*64*5*5];
  static float bias3[1024];
  static float weight3[7*7*64*1024];
  static float bias4[10];
  static float weight4[1024*10]; 



  read_bias1("weight_data/bias1.bin", 32, bias1);
  read_weight1("weight_data/weight1.bin", 32*5*5, weight1);


  read_bias1("weight_data/bias2.bin", 64, bias2);
  read_weight1("weight_data/weight2.bin", 32*64*5*5, weight2);


  read_bias1("weight_data/bias3.bin", 1024, bias3);
  read_weight1("weight_data/weight3.bin", 7*7*64*1024, weight3);


  read_bias1("weight_data/bias4.bin", 10, bias4);
  read_weight1("weight_data/weight4.bin", 1024*10, weight4);


  //compute input name
 
  // sprintf(file_path,"MNIST_images/image0.bin");
  sprintf(file_path, argv[1]);

  read_mnist(file_path, 784, in_image);

        
  //perform feed forward operation thourgh the network
  run_convolution_layer1(in_image, net_layer1, bias1, weight1);
  run_convolution_layer2(net_layer1, net_layer2, bias2, weight2);
  run_convolution_layer3(net_layer2, net_layer3, bias3, weight3);
  run_convolution_layer4(net_layer3, bias4, weight4, probabilities); 


  int result=-1; 
  float max_number=0;
  for(i=0;i<10;i++){
  	if(probabilities[i]>max_number){
  		result = i;
  		max_number = probabilities[i];
  	}
  	printf("The probabilities for %d is %f\n", i, probabilities[i]);
  }
  printf(" The classification result is %d\n",result);
    
  return 0;
}
