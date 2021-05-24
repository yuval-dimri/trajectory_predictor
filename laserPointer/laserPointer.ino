#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVO_COMPUTERN_X 0
#define SERVO_COMPUTERN_Y 1
#define COMPUTER_SERIAL Serial
#define DATA_FRESHNESS_TIME 100
int servoCalibration[] = {150, 600, 150, 400}; //us min max, min max
// String availableDataTypes = "pd";                //all kind of data types: p(position), d(degrees )
int degreesArray[] = {0, 0};
long dataRecivedTime = 100000;

void setup()
{
    COMPUTER_SERIAL.begin(115200);
    while (!Serial)
    {
        ; // wait for serial port to connect. Needed for native USB port only
    }
    pwm.begin();
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(50); //Analog servos run at ~50 Hz updates
    delay(10);
}

void loop()
{
    updateComputerData('d');
    if (Serial.available() > 0)
        COMPUTER_SERIAL.write(degreesArray[0]);
    moveServo(0, degreesArray[0]);
    moveServo(1, degreesArray[1]);
}

void moveServo(int servo, int deg)
{
    int pulseLen = map(deg, 0, 180, servoCalibration[servo * 2], servoCalibration[servo * 2 + 1]);
    //    pwm.writeMicroseconds(servo, pulseLen);
    pwm.setPWM(servo, 0, pulseLen);

    // updateComputerData('d');
}

String readDataFromComputer()
{
    String packetRecieved = "";
    while (COMPUTER_SERIAL.available() > 0)
    {
        if (COMPUTER_SERIAL.read() == '!')
        {
            packetRecieved = COMPUTER_SERIAL.readStringUntil('\n');
            //      Serial.println(packetRecieved);
            return packetRecieved;
        }
    }
    //Serial.println("cant recive data from mySerial");
    return "";
}

bool parseIncomingData(String incomingString, char dataType, int *posArr)
{
    String tempString = "";
    char ch;
    bool startRead = false;
    uint8_t valueCounter = 0;
    //  valueCounter = 0;

    //  Serial.println(incomingString.length());

    for (int i = 0; i < incomingString.length(); i++)
    {
        ch = incomingString.charAt(i);
        //Serial.print(i);
        if (startRead)
        {
            if (isDigit(ch) || ch == '-')
            {
                tempString.concat(ch);
                //Serial.printf("i = %d, tempString = %d \n",i,tempString);
            }
            else if (ch == ',')
            {
                posArr[valueCounter] = tempString.toInt();
                //                Serial.print(posArr[valueCounter]);
                //                Serial.print(", ");
                tempString = "";
                valueCounter++;
            }
            else
            {
                startRead = false;
            }
        }

        if (ch == dataType && startRead == false)
        {
            startRead = true;
            valueCounter = 0;
            //Serial.println("dataKind");
        }
    }
    // Serial.println();
}

bool updateComputerData(char dataTypeForParsing)
{
    String rawDataRecived = "";
    rawDataRecived = readDataFromComputer(); //get raw data from rCOMPUTER

    if (rawDataRecived.length() > 0)
    {
        parseIncomingData(rawDataRecived, dataTypeForParsing, degreesArray); //get position data into positionArray = [x,y,rot]
        dataRecivedTime = millis();
    }
    return (millis() - dataRecivedTime) < DATA_FRESHNESS_TIME; //data fresh(true) or not(false)
}