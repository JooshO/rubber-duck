data_dict = {}
NAME_KEY = "name"
LANG_KEY = "lang"



'''
Currently just echoes the user's input
Consider: https://github.com/openai/gpt-2
'''
def main():
    data_dict[NAME_KEY] = input("Hello! What is your name?\n")
    data_dict[LANG_KEY] = input(f"Hello {data_dict[NAME_KEY]}, what language or framework are you working in today?\n")

    print("Alright, I am ready to listen!")
    finished = False
    while not finished:
        user_in = input().lower()

        # processing here!
        if "quit" in user_in or "exit" in user_in:
            finished = True
            break

        print(user_in)

    print("Good bye!")

if __name__ == "__main__":
    main()