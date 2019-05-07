#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>

// written by M. Alexander Crim 
// spring 19' 

using namespace std;

struct dataSet{
    float    x,y,b,
            target[2];
    dataSet(){}
};

struct neuralNetwork{
    float	outputNodes[2],
            intermediateNodes[3],
            inputNodes[3],
            inputToIntermediate[3][3],//Rows are the input nodes columns are intermediate nodes. weights are in the value at [input][intermediate]
            intermediateToOutput[3][2];//rows are intermediate and columns are output. weights are at [intermediate][output]
			
}neuralNetwork;


float randFloat(float min, float max){

    float random = ((float) rand())/ (float)RAND_MAX;
    float range = max-min;
    return random*range+min;
}

void printNetwork(struct neuralNetwork *network_ptr){
	
	cout << "--------Network---------" << endl;
    cout << network_ptr->inputNodes[0] <<"  " << network_ptr->intermediateNodes[0] << "   " << network_ptr->outputNodes[0] << endl;
    cout << network_ptr->inputNodes[1] <<"  " << network_ptr->intermediateNodes[1] << "   " << network_ptr->outputNodes[1] << endl;
    cout << network_ptr->inputNodes[2] <<"  " << network_ptr->intermediateNodes[2] << endl;

}

void printWeights(struct neuralNetwork* network_ptr){

	cout << "------------input to intermediate ------------" <<endl;
    for(int i=0; i<3;i++){ //only prints weights between input and intermediate
        for(int j=0; j<3; j++){
            cout << network_ptr->inputToIntermediate[i][j] << "    ";
        }
        cout << endl;
    }

	cout << "------------intermediate to output------------" <<endl;
    for(int i=0; i<3;i++){ //only prints weights between input and intermediate
        for(int j=0; j<2; j++){
            cout << network_ptr->intermediateToOutput[i][j] << "    ";
        }
        cout << endl;
    }
	cout << endl<< endl;
}

//sets all edges to a random weight
void initalization(struct neuralNetwork* network_ptr){

    for(int i=0; i<3; i++)
        for(int j = 0; j<3;j++)
            network_ptr->inputToIntermediate[i][j] = randFloat(-1,1);

    for(int i=0; i<3; i++)
        for(int j=0; j<2; j++)
            network_ptr->intermediateToOutput[i][j] = randFloat(-1,1);
}

float activationFunction(float net){
    return (1/(1 +  pow(M_E, -net)));
}

void assignNodeValues(struct neuralNetwork* network_ptr){
    //assigns intermediate nodes' values
    float net = 0;
    for(int i = 0; i < 3; i++){
        for(int j=0; j<3; j++){//this innerloops gets the net activation for each node
            net+= network_ptr->inputNodes[j]*network_ptr->inputToIntermediate[j][i];
        }
        network_ptr->intermediateNodes[i] = activationFunction(net); //puts the net into the activation function
        net = 0;
    }
    //assign output nodes values'
    net =0;
    for(int i=0; i< 2; i++){
        for(int j=0; j<3; j++)
            net += network_ptr->intermediateNodes[j]*network_ptr->intermediateToOutput[j][i];
        network_ptr->outputNodes[i]  = activationFunction(net);
        net = 0;
    }
}


//Backpropagation function
void adjustWeights(struct neuralNetwork* network_ptr, struct dataSet data){
    float stepsize = 0.1,
            summation=0,
            deltaW=0;

    for(int i=0; i<2; i++){//adjust weights between intermediate and output
        for(int j=0; j<3; j++){
            deltaW =stepsize*(network_ptr->outputNodes[i] - data.target[i]) * network_ptr->outputNodes[i]* (1 -network_ptr->outputNodes[i]) * network_ptr->intermediateNodes[j];
            //cout << "weight: " << network_ptr->intermediateToOutput[j][i] << "  deltaW:  " << deltaW;
            network_ptr->intermediateToOutput[j][i] -= deltaW;

			if(network_ptr->intermediateToOutput[j][i] > 1) network_ptr->intermediateToOutput[j][i] =1;
			if(network_ptr->intermediateToOutput[j][i] <-1) network_ptr->intermediateToOutput[j][i] =-1;
            //cout << "  new weight: " << network_ptr->intermediateToOutput[j][i] << endl;
        }
    }

    for(int i=0;i<3; i++){ //adjust weights between input and intermediate
        for(int j=0;j<3; j++){
            for(int k=0; k<2; k++)//this does the summation portion of the weight adjusment.
                summation =(network_ptr->outputNodes[k] - data.target[k]) * network_ptr->outputNodes[k]* (1 -network_ptr->outputNodes[k]) * network_ptr->intermediateNodes[j]*network_ptr->intermediateToOutput[j][k]
                           * network_ptr->intermediateNodes[j]* (1- network_ptr->intermediateNodes[j]) * network_ptr->inputNodes[i];

            network_ptr->inputToIntermediate[j][i] -= stepsize*summation;
			//take these out to let  weightsget out of bounds of [-1,1] . This results in a much faster training time.
			//if(network_ptr->inputToIntermediate[j][i] > 1) network_ptr->inputToIntermediate[j][i] =1;
			//if(network_ptr->inputToIntermediate[j][i] <-1) network_ptr->inputToIntermediate[j][i] =-1;
        }
    }
}

void trainNetwork(struct neuralNetwork* network_ptr, struct dataSet trainingSet[50]){
    int successes=0,
            it=0;
    while(!(successes == 4)){ //ends once there is a 100% complete.
        successes=0;
        it++;

        for(int i=0; i < 4; i++){
            //take inputs from dataSet[i] and place them into the input nodes
            network_ptr->inputNodes[0] = trainingSet[i].x;
            network_ptr->inputNodes[1] = trainingSet[i].y;
            network_ptr->inputNodes[2] = trainingSet[i].b;
            assignNodeValues(network_ptr); //generates the values of all nodes
			
			if(it ==0)
				printNetwork(network_ptr);

			//sigmoidal activation so it can never output 0 so used round to test.
            if(trainingSet[i].target[0] == round(network_ptr->outputNodes[0]) && trainingSet[i].target[1] == round(network_ptr->outputNodes[1]))
                successes++;
			else
            	adjustWeights(network_ptr, trainingSet[i]);
		}
		it++;
    }
    cout<< "Trained in " << it << " iterations." << endl;
}

int main(int argc, char *argv[]){

    struct neuralNetwork myNetwork;
    struct dataSet trainingSet[4];

	//Hindsight being 20/20, If i was going to write this neural network better I would have done it with 2 hidden layers. Layer h1 with 2 neurons and the h2 with one neuron.
	//because this would better represent the relationship between x and y in  a xor gate made with the basic 3 gates. !(x&&y) && (x||y) 
	
	//train xor. 
	trainingSet[0].x = 1;//X
	trainingSet[0].y =1;  // Y
	trainingSet[0].b = 1; //bias
	trainingSet[0].target[0] = 0;
	trainingSet[0].target[1] = 0;

	trainingSet[1].x = 1;//X
	trainingSet[1].y =0;  // Y
	trainingSet[1].b = 1; //bias
	trainingSet[1].target[0] = 1;
	trainingSet[1].target[1] = 1;

	trainingSet[2].x = 0;//X
	trainingSet[2].y =1;  // Y
	trainingSet[2].b = 1; //bias
	trainingSet[2].target[0] = 1;
	trainingSet[2].target[1] = 1;

	trainingSet[3].x = 0;//X
	trainingSet[3].y =0;  // Y
	trainingSet[3].b = 1; //bias
	trainingSet[3].target[0] = 0;
	trainingSet[3].target[1] = 0;
	

    initalization(&myNetwork);
	cout << "---------Initial Weights---------" << endl;
	printWeights(&myNetwork);

    trainNetwork(&myNetwork, trainingSet);
	
	cout << "---------Final Weights---------" << endl;
	printWeights(&myNetwork);

    return 0;
}
