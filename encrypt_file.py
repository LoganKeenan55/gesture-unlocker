import os
from cryptography.fernet import Fernet

### 

file_path = "H:\Videos\Hand\code\hand-gesture-password\Secret.txt"

###

key = Fernet.generate_key()
os.system('cls')
print("Here is the code: " + key.decode() + " keep it safe!")

fernet = Fernet(key)

file = open(file_path, "rb")
original_data = file.read()
file.close

encrypted_data = fernet.encrypt(original_data)

file = open("Secret.enc", "wb")
file.write(encrypted_data)
file.close


print("Encryption complete! You can now delete the original file.")