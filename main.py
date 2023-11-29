import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path>")
        sys.exit(1)

    path = sys.argv[1]
    print(path)


if __name__ == "__main__":
    main()
