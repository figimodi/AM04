import pyautogui
import time
import keyboard

def perform_hotkey_sequence_1():
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'shift', 'n')
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'i')
    time.sleep(0.1)
    pyautogui.press('f')
    time.sleep(0.1)
    pyautogui.keyDown('shift')
    pyautogui.leftClick()
    time.sleep(0.2)
    pyautogui.keyUp('shift')
    time.sleep(0.1)
    pyautogui.press('F4')

def perform_hotkey_sequence_2():
    pyautogui.hotkey('alt', 'shift', 'x')
    pyautogui.hotkey('alt', 'shift', 'y')

def main():
    print("Listening for key presses... (Press 'esc' to exit)")

    # Define the key events and their corresponding actions
    keyboard.add_hotkey('/', perform_hotkey_sequence_1)
    keyboard.add_hotkey('f2', perform_hotkey_sequence_2)

    # Keep the script running and listening for events
    keyboard.wait('esc')  # Stop the script when 'esc' is pressed

if __name__ == "__main__":
    main()