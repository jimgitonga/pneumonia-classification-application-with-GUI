import urllib
import requests

def book_to_words(book_url='https://www.gutenberg.org/files/84/84-0.txt'):
    booktxt = urllib.request.urlopen(book_url).read().decode()
    bookascii = booktxt.encode('ascii','replace')
    bookascii=bookascii.split()
    return bookascii

def radix_a_book(book_url='https://www.gutenberg.org/files/84/84-0.txt'):
    
    # Creatin a radixsort Function
    # x = book_to_words()
    maxl = len(max(book_to_words()))
    print(maxl)

    # Compute and Find Largest element from the predifined list
    # for i in x:
    #     if max < i:
    #         max = i


    # Perorm Counting sort  based on position.
    position = 1
    while maxl / position > 0:
        countSort(book_to_words(), position)
        position *= 10

def countSort(book_url, position):
    x = len(book_url)
    output = [0 for i in range(0, x)]

    # range of the number is 0-9 for each position  considered.
    frequency = [0 for i in range(0, 10)]

    # Calculate number of occurrences in frequency
    for i in range(0, x):
        frequency[(len(book_url[i]) // position) % 10] += 1

    # Increment count[i] to contains actual position of the new digit in output[]
    for i in range(1, 10):
        frequency[i] += frequency[i - 1]

        # BCompute output
    for i in range(x - 1, -1, -1):
        output[frequency[(len(book_url[i]) // position) % 10] - 1] = len(book_url[i])
        frequency[(len(book_url[i]) // position) % 10] -= 1

    # Copy the output array to the input Array, This makes the sorted array appeear
    for i in range(0, x):
        book_url[i] = output[i]
        print("sorted {}".format(i))

    # print a list

def PrintList(book_url):
    for i in book_url:
        print(i, end=" ")
    print("\n")

# executing and testing code

radix_a_book()
print("Radix Sorted List")
PrintList(book_to_words())
# book_to_words()