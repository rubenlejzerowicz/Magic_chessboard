#include <QueueArray.h>

#define dirPinA 13
#define stepPinA 12

#define dirPinB 11
#define stepPinB 10

#define switchX 40
#define switchY 42
#define magnetPin 44

#define sSteps 150
#define dSteps 300

int speedM = 1300;

struct chessMove
{
   int dir;
   int mg;
   
};

bool transmission = true;

QueueArray <chessMove> moves;

void Ac(int steps){
  digitalWrite(dirPinA, LOW);
  for (int i = 0; i < steps; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPinA, HIGH);
    delayMicroseconds(speedM);
    digitalWrite(stepPinA, LOW);
    delayMicroseconds(speedM);
  }
}
void Bc(int steps){
  digitalWrite(dirPinB, LOW);
  for (int i = 0; i < steps; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPinB, HIGH);
    delayMicroseconds(speedM);
    digitalWrite(stepPinB, LOW);
    delayMicroseconds(speedM);
  }
}

void Aac(int steps){
  digitalWrite(dirPinA, HIGH);
  for (int i = 0; i < steps; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPinA, HIGH);
    delayMicroseconds(speedM);
    digitalWrite(stepPinA, LOW);
    delayMicroseconds(speedM);
  }
}

void Bac(int steps){
  digitalWrite(dirPinB, HIGH);
  for (int i = 0; i < steps; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPinB, HIGH);
    delayMicroseconds(speedM);
    digitalWrite(stepPinB, LOW);
    delayMicroseconds(speedM);
  }
}

void AcBc(int steps){
  digitalWrite(dirPinA, LOW);
  digitalWrite(dirPinB, LOW);
  // Spin the stepper motor 1 revolution slowly:
  for (int i = 0; i < steps; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPinA, HIGH);
    digitalWrite(stepPinB, HIGH);
    delayMicroseconds(speedM);
    digitalWrite(stepPinA, LOW);
    digitalWrite(stepPinB, LOW);
    delayMicroseconds(speedM);
  }
  
}

void AcBac(int steps){
  digitalWrite(dirPinA, LOW);
  digitalWrite(dirPinB, HIGH);
  for (int i = 0; i < steps; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPinA, HIGH);
    digitalWrite(stepPinB, HIGH);
    delayMicroseconds(speedM);
    digitalWrite(stepPinA, LOW);
    digitalWrite(stepPinB, LOW);
    delayMicroseconds(speedM);
  }

  
}

void AacBc(int steps){
  digitalWrite(dirPinA, HIGH);
  digitalWrite(dirPinB, LOW);
  for (int i = 0; i < steps; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPinA, HIGH);
    digitalWrite(stepPinB, HIGH);
    delayMicroseconds(speedM);
    digitalWrite(stepPinA, LOW);
    digitalWrite(stepPinB, LOW);
    delayMicroseconds(speedM);
  }

  
}

void AacBac(int steps){
  digitalWrite(dirPinA, HIGH);
  digitalWrite(dirPinB, HIGH);
  for (int i = 0; i < steps; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPinA, HIGH);
    digitalWrite(stepPinB, HIGH);
    delayMicroseconds(speedM);
    digitalWrite(stepPinA, LOW);
    digitalWrite(stepPinB, LOW);
    delayMicroseconds(speedM);
  }
  
}

void reset() {
  while (digitalRead(switchX) != LOW){
    AcBc(1);
  }

  AacBac(30);
  delay(100);

  while (digitalRead(switchY) != LOW){
    AcBac(1);
  }
  AacBc(15);
  delay(100);
  
}

void activateMagnet(){
  digitalWrite(magnetPin, HIGH);
}

void deactivateMagnet(){
  digitalWrite(magnetPin, LOW);
}

void setup() {

  // Declare pins of stepper motors and electromagnet as output:

  Serial.begin(57600);
  pinMode(stepPinA, OUTPUT);
  pinMode(dirPinA, OUTPUT);
  pinMode(stepPinB, OUTPUT);
  pinMode(dirPinB, OUTPUT);
  pinMode(magnetPin, OUTPUT);

  pinMode(switchX, INPUT);
  pinMode(switchY, INPUT);

  
  delay(2000);
  
  reset();

  

}

void loop() {
  // put your main code here, to run repeatedly:
  
  if (transmission){ 
    while (Serial.available() > 0) {
      String myString = Serial.readStringUntil('\n');
      int commaIndex = myString.indexOf(',');
      //  Search for the next comma just after the first
      
  
      String firstValue = myString.substring(0, commaIndex);
      String secondValue = myString.substring(commaIndex + 1);
      
  
      int firstVal = firstValue.toInt();
      int secondVal = secondValue.toInt();
      
      if (firstVal == 10){
        transmission = false;
      }
      else {
        chessMove m;
        m.dir = firstVal;
        m.mg = secondVal;
        moves.enqueue(m);
      }
      
    }
  }
    

    
  
  else {
    
  
    while (!moves.isEmpty()){
  
      chessMove mov = moves.dequeue();
  
      if (mov.mg == 1){
        activateMagnet();
      }
  
      if (mov.mg == 0){
        deactivateMagnet();
      }
  
      if (mov.dir == 1){
        Bc(dSteps);
      }
  
      if (mov.dir == 2){
        AacBc(sSteps);
      }
      
      if (mov.dir == 3){
        Aac(dSteps);
      }
  
      if (mov.dir == 4){
        AcBc(sSteps);
      }
  
      if (mov.dir == 6){
        AacBac(sSteps);
      }
  
      if (mov.dir == 7){
        Ac(dSteps);
      }
  
      if (mov.dir == 8){
        AcBac(sSteps);
      }
  
      if (mov.dir == 9){
        Bac(dSteps);
      }

      if (moves.isEmpty()){
        deactivateMagnet();
      }
  
      
      
  
      
      
    }
    transmission = true;
  }
  
  

}
