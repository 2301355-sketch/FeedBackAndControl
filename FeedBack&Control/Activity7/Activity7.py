import serial
import time

def run_servo_controller(port, baud_rate=9600):
    try:
        # Create a serial connection
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Connecting to Arduino on {port}...")
        time.sleep(2)  # Wait for Arduino to reset
        ser.reset_input_buffer()
        print("Connection established. Type 'exit' to quit.")

        while True:
            user_input = input("\nEnter angle (0-180): ").strip().lower()
            if user_input == 'exit':
                break
            
            # Process user input for angle
            try:
                angle = int(user_input)
                
                if 0 <= angle <= 180:
                    # Send angle to Arduino
                    ser.write(f"{angle}\n".encode('utf-8'))
                    time.sleep(0.3)  # Wait for Arduino to process
                    
                    # Check for Arduino response
                    if ser.in_waiting > 0:
                        feedback = ser.readline().decode('utf-8').strip()
                        print(feedback)  # Print confirmation from Arduino
                    else:
                        print("Waiting for Arduino response...")
                else:
                    print("Error: Range is 0-180.")
            except ValueError:
                print("Error: Please enter a valid number.")

    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    finally:
        # Ensure the serial port is closed on exit
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Port closed.")

if __name__ == "__main__":
    run_servo_controller(port="COM6")
