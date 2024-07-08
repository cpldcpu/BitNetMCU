size = 16
threshold = 0x80
threshold_divider = threshold

def upscale_enlarge(input_image = None, output_size = size):
    # upscale data from 8x8 to output_size x output_size
    input_size = int(output_size / 2)
    output_image = [ (1-input_image[(y//2) * input_size + (x//2)])*threshold for y in range(output_size) for x in range(output_size) ]
    return output_image

def upscale_half_greyscale(input_image, output_size = size):
    input_size = int(len(input_image)**0.5)
    output_image = [threshold for i in range(output_size * output_size)]
    for y in range(output_size):
        for x in range(output_size):
            # apply half greyscale to nearest 4 pixels
            input_x = x // (output_size // input_size)
            input_y = y // (output_size // input_size)
            if input_image[input_y * input_size + input_x] and x < output_size - 1 and y < output_size - 1:
                for dy in [-1,1]:
                    for dx in [-1,1]:
                        output_image[(y + dy) * output_size + (x + dx)] = threshold // threshold_divider

    for y in range(output_size):
        for x in range(output_size):
            input_x = x // (output_size // input_size)
            input_y = y // (output_size // input_size)
            if input_image[input_y * input_size + input_x]:
                output_image[y * output_size + x] = threshold - 1
    return output_image


data = [0,0,0,0,0,0,0,0,
        0,0,1,1,1,1,0,0,
        0,0,0,0,0,1,0,0,
        0,0,0,0,0,1,0,0,
        0,0,1,1,1,1,0,0,
        0,0,1,0,0,0,0,0,
        0,0,1,1,1,1,0,0,
        0,0,0,0,0,0,0,0]

data = upscale_enlarge(data)
# 0xEC = 11101100
print('\t\t\t\tThreshold: %d, 0x%X' % (threshold, threshold))
for y in range(size):
    for x in range(size):
        if data[y * size + x] >= threshold:
            print(' ', end='')
        elif data[y * size + x] <= threshold // threshold_divider:
            print('.', end='')
        else:
            print('1', end='')
    print(' '*6, end='')
    for x in range(size):
        if data[y * size + x] >= threshold:
            print(f'\033[91m-{str(abs(data[y * size + x] - threshold * 2)).zfill(3)}\033[0m', end=' ')
        else:
            print(f'+{str(data[y * size + x]).zfill(3)}', end=' ')
    print()

for i in data:
    if (i>=threshold):
        print(i-threshold*2, end=', ')
    else:
        print(i, end=', ')
