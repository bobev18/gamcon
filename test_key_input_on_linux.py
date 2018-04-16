import time
import random
from pynput.keyboard import Key, Controller

keyboard = Controller()

def tap(key):
	keyboard.press(key)
	keyboard.release(key)


# Press and release space
# keyboard.press(Key.space)
# keyboard.release(Key.space)

# # Type a lower case A; this will work even if no key on the
# # physical keyboard is labelled 'A'
# keyboard.press('a')
# keyboard.release('a')

# # Type two upper case As
# keyboard.press('A')
# keyboard.release('A')
# with keyboard.pressed(Key.shift):
#     keyboard.press('a')
#     keyboard.release('a')

# # Type 'Hello World' using the shortcut type method
# keyboard.type('Hello World')


keylist = ['w', 'a', 's', 'd']

print('get game window in focus')
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)


for i in range(50):
	tap(random.choice(keylist))