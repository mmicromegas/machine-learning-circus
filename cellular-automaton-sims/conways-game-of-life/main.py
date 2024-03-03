# https://www.youtube.com/watch?v=cRWg2SWuXtM
import sys
import time
import pygame
import numpy as np

COLOR_BG = (10, 10, 10)
COLOR_GRID = (40, 40, 40)
COLOR_DIE_NEXT = (170, 170, 170)
COLOR_ALIVE_NEXT = (255, 255, 255)


def update(screen, cells, size, with_progress=False):
    updated_cells = np.zeros((cells.shape[0], cells.shape[1]))

    for row, col in np.ndindex(cells.shape):
        alive = np.sum(cells[row - 1:row + 2, col-1:col+2]) - cells[row, col]
        color = COLOR_BG if cells[row, col] == 0 else COLOR_ALIVE_NEXT

        if cells[row, col] == 1:
            if alive < 2 or alive > 3:  # die due to starvation or overpopulation
                if with_progress:
                    color = COLOR_DIE_NEXT
                    #print(row, col, alive, color, "die due to starvation or overpopulation")
            elif 2 <= alive <= 3:  # we survive to the next generation
                updated_cells[row, col] = 1
                if with_progress:
                    color = COLOR_ALIVE_NEXT
                    #print(row, col, alive, color, "survive to the next generation")
        else:
            if alive == 3:
                updated_cells[row, col] = 1
                if with_progress:
                    color = COLOR_ALIVE_NEXT
                    #print(row, col, alive, color, "born")

        # draw rectangle
        pygame.draw.rect(screen, color, (col * size, row * size, size - 1, size - 1))

    return updated_cells

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 800))

    cells = np.zeros((80, 80))

    # insert ones to cells array into a circle
    #for i in range(0, 60):
    #    for j in range(0, 80):
    #        if (i - 30) ** 2 + (j - 40) ** 2 < 400:
    #            cells[i, j] = 1


    # read png file and convert it to numpy array
    img = pygame.image.load('NESS_logo_download.png')
    #img = pygame.image.load('actimize_logo.png')

    # resize arr to 80x60
    arr = pygame.transform.scale(img, (80, 80))
    arr = pygame.surfarray.array2d(arr)
    # rotate arr by 90 degrees and flip sides
    arr = np.rot90(arr)
    arr = np.rot90(arr)
    arr = np.rot90(arr)
    arr = np.fliplr(arr)

    # change 1 to 0 and 0 to 1
    arr = np.where(arr == 1, 1, 0)


    cells = arr

    screen.fill(COLOR_GRID)
    update(screen, cells, 10)

    pygame.display.flip()
    pygame.display.update()

    running = False

    # game loop

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = not running
                    update(screen, cells, 10)
                    pygame.display.update()
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                cells[pos[1] // 10, pos[0] // 10] = 1
                update(screen, cells, 10)
                pygame.display.update()

        screen.fill(COLOR_GRID)

        # actual simulations
        if running:
            cells = update(screen, cells, 10, with_progress=True)
            pygame.display.update()
            #sys.exit()

        time.sleep(0.001)


if __name__ == '__main__':
    main()
