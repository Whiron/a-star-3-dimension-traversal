"""
CSCI 630 : Foundations of Artificial Intelligence
Lab 1
Author: Sarthak Thakkar (st4070)

lab1.py

This Program takes 5 input from users
Terrain file,Elevation data, Path file, Season, Output file
and processes the path to trave lall the given nodes with
A* search algorithm. to fine path with least time from start
of the course to end path and writes it to an image and
shows user the outputi n human readable format.
"""

import sys
from PIL import Image
import numpy as np
import math


"""
This method loads Co-ordinates from given path file

:param  path_file   path of file 
"""
def load_path_file(path_file):
    print("loading path files from : ",path_file)
    file1 = open(path_file)
    row=0
    for l in file1:
        row+=1
    file = open(path_file)
    rows,cols=(row,2)
    arr = [[0 for t in range(cols)] for j in range(rows)]

    t = 0
    for line in file:
        line = line.lstrip()

        contents = line.split(' ')
        j=0
        for word in contents:
            arr[t][j]=int(word)
            j+=1
        t+=1

    return arr,row

"""
This method loads elevation data from given elevation file

:param  elevation_file   elevation data file 
"""
def load_elevation_file(elevation_file):
    print("loading elevation file from : ", elevation_file)
    rows,cols=(500,375)
    arr = [[0 for i in range(cols)] for j in range(rows)]
    file = open(elevation_file)

    i=0
    for line in file:
        line=line.lstrip()
        contents=line.split('   ')
        j=0
        for word in contents:
            if(j<375):
                arr[i][j]=float(word)
            j+=1
        i+=1

    arr= np.array(arr).T.tolist()
    return arr
"""
This method loads Co-ordinates from given terrain image file

:param  terrain_file    path of terrain file
"""
def load_terrain_file(terrain_file):
    print("loading image file from : ",terrain_file)
    rows, cols = (375,500)
    arr = [[0 for i in range(cols)] for j in range(rows)]
    img = Image.open(terrain_file)
    pix = img.load()


    for i in range(375):
        for j in range(500):
            value = pix[i, j][0:3]

            if (value == (205, 0, 101)):
                legend_value = 'O'
            if (value == (0, 0, 0)):
                legend_value = 'M'
            if (value == (71, 51, 3)):
                legend_value = 'K'
            if (value == (0, 0, 255)):
                legend_value = 'H'
            if (value == (5, 73, 24)):
                legend_value = 'G'
            if (value == (2, 136, 40)):
                legend_value = 'F'
            if (value == (2, 208, 60)):
                legend_value = 'E'
            if (value == (255, 255, 255)):
                legend_value = 'C'
            if (value == (255, 192, 0)):
                legend_value = 'B'
            if (value == (248, 148, 18)):
                legend_value = 'A'

            arr[i][j] = legend_value

    return arr

"""
This method creates a graph of neighbour nodes as per the their co-ordinates

:param  elevation_file   path of elevation file  
"""
def make_neighbours(elevation_data):
    rows, cols = (375, 500)
    arr = [[0 for i in range(cols)] for j in range(rows)]
    for i in range(375):
        for j in range(500):
            if(i==0):
                if(j==0):
                    neighbours = []
                    neighbours.append([i+1,j,elevation_data[i+1][j]])
                    neighbours.append([i+1,j+1,elevation_data[i + 1][j+1]])
                    neighbours.append([i,j+1,elevation_data[i][j+1]])
                    arr[i][j]=neighbours
                elif(j==499):
                    neighbours = []
                    neighbours.append([i+1,j,elevation_data[i + 1][j]])
                    neighbours.append([i+1,j-1,elevation_data[i + 1][j - 1]])
                    neighbours.append([i,j-1,elevation_data[i][j - 1]])
                    arr[i][j] = neighbours
                else:
                    neighbours = []
                    neighbours.append([i+1,j,elevation_data[i + 1][j]])
                    neighbours.append([i,j-1,elevation_data[i][j - 1]])
                    neighbours.append([i,j+1,elevation_data[i][j + 1]])
                    arr[i][j] = neighbours

            elif(i==374):
                if (j == 0):
                    neighbours = []
                    neighbours.append([i-1,j,elevation_data[i - 1][j]])
                    neighbours.append([i-1,j-1,elevation_data[i - 1][j - 1]])
                    neighbours.append([i,j-1,elevation_data[i][j - 1]])
                    arr[i][j] = neighbours
                elif (j == 499):
                    neighbours = []
                    neighbours.append([i-1,j,elevation_data[i - 1][j]])
                    neighbours.append([i-1,j-1,elevation_data[i - 1][j - 1]])
                    neighbours.append([i,j-1,elevation_data[i][j - 1]])
                    arr[i][j] = neighbours
                else:
                    neighbours = []
                    neighbours.append([i-1,j,elevation_data[i - 1][j]])
                    neighbours.append([i,j-1,elevation_data[i][j - 1]])
                    neighbours.append([i,j+1,elevation_data[i][j + 1]])
                    arr[i][j] = neighbours
            elif(j==0):
                neighbours = []
                neighbours.append([i-1,j,elevation_data[i - 1][j]])
                neighbours.append([i,j+1,elevation_data[i][j + 1]])
                neighbours.append([i+1,j,elevation_data[i+1][j]])
                arr[i][j] = neighbours
            elif (j == 499):
                neighbours = []
                neighbours.append([i-1,j,elevation_data[i - 1][j]])
                neighbours.append([i,j-1,elevation_data[i][j - 1]])
                neighbours.append([i-1,j-1,elevation_data[i + 1][j - 1]])
                arr[i][j] = neighbours

            else:
                neighbours = []
                neighbours.append([i-1,j,elevation_data[i - 1][j]])
                neighbours.append([i+1,j,elevation_data[i + 1][j]])
                neighbours.append([i,j+1,elevation_data[i][j + 1]])
                neighbours.append([i,j-1,elevation_data[i][j - 1]])
                neighbours.append([i+1,j+1,elevation_data[i + 1][j + 1]])
                neighbours.append([i+1,j-1,elevation_data[i + 1][j - 1]])
                neighbours.append([i-1,j-1,elevation_data[i - 1][j - 1]])
                neighbours.append([i-1,j+1,elevation_data[i - 1][j + 1]])
                arr[i][j] = neighbours

    return arr

"""
This method generates ice path and freezes the water bodies upto 7 pixels
from the bank using BFS approach

:param  merged_elevation_data   List of neighbours of every node
:param  terrain_data            List of Terrain type of every co-ordinate 
"""

def make_ice_path(merged_elevation_data, terrain_data):
    print("making ice path")
    winter_terrain_data=terrain_data
    bank_water=[]
    ice_data=[]
    queue=[]
    for i in range(375):
        for j in range(500):
            if terrain_data[i][j]=='H':
                flag1=False
                for neighbour in merged_elevation_data[i][j]:
                    if (terrain_data[neighbour[0]][neighbour[1]]) != 'H' and (terrain_data[neighbour[0]][neighbour[1]]) != 'P':
                        flag1=True
                if(flag1==True):
                    ice_data.append([i,j])
                    bank_water.append([i,j])    # All nodes on bank of water bodies

    tmpQueue = []
    for node in bank_water:
        queue.clear()
        tmpQueue.clear()
        queue.append(node)
        i=0
        while i<6:
            while len(queue) > 0:
                tempNode=queue.pop()

                neighbours = merged_elevation_data[tempNode[0]][tempNode[1]]
                for neighbour in neighbours:
                    if terrain_data[neighbour[0]][neighbour[1]] == 'H' and [neighbour[0],neighbour[1]] not in ice_data:
                        tmpQueue.append(neighbour)
                        ice_data.append([neighbour[0],neighbour[1]])

            if(len(queue)==0):
                queue=tmpQueue.copy()
                tmpQueue.clear()
                i+=1

    for node in ice_data:
        winter_terrain_data[node[0]][node[1]]= 'P'

    return ice_data,winter_terrain_data

"""
This method generates mud path and swamps the land around water bodies 
upto 7 pixels from the bank using BFS approach

:param  elevation_data          List of elevation for all nodes
:param  merged_elevation_data   List of neighbours of every node
:param  terrain_data            List of Terrain type of every co-ordinate
"""

def make_mud_path(elevation_data,merged_elevation_data, terrain_data):
    print("making mud path")
    spring_terrain_data=terrain_data
    bank_water=[]
    mud_data=[]
    queue=[]
    for i in range(375):
        for j in range(500):
            if terrain_data[i][j]=='H':
                flag1=False
                for neighbour in merged_elevation_data[i][j]:
                    if (terrain_data[neighbour[0]][neighbour[1]]) != 'H':
                        flag1=True
                if(flag1==True):
                    bank_water.append([i,j])    # All nodes on bank of water bodies

    cqueue=[]
    for node in bank_water:
        queue.clear()
        cqueue.clear()
        baseElevation = elevation_data[node[0]][node[1]]
        queue.append(node)
        i=0
        while i<15:
            while len(queue) >0:
                tempNode=queue.pop()
                neighbours = merged_elevation_data[tempNode[0]][tempNode[1]]
                for neighbour in neighbours:
                    current_elevation=elevation_data[neighbour[0]][neighbour[1]]
                    tempDistance = (baseElevation-current_elevation)
                    if terrain_data[neighbour[0]][neighbour[1]] != 'H' and terrain_data[neighbour[0]][neighbour[1]] != 'O' and [neighbour[0],neighbour[1]] not in mud_data and tempDistance < 1:
                        cqueue.append(neighbour)
                        mud_data.append([neighbour[0],neighbour[1]])

            if(len(queue) == 0):
                queue=cqueue.copy()
                cqueue.clear()
                i+=1

    for node in mud_data:
        spring_terrain_data[node[0]][node[1]]= 'Q'

    return mud_data,spring_terrain_data

"""
This method Generates Speed Dictionary as per season

:param  season  The season for accessing speed of terrains
"""
def generate_speed_dictionary(season):
    dict={}
    if season=="summer":
        dict['A'] = 2.2352
        dict['B'] = 1.78816
        dict['C'] = 2.0120
        dict['E'] = 1.6412
        dict['F'] = 1.7956
        dict['G'] = 1.0234
        dict['H'] = 0.5210
        dict['K'] = 2.68224
        dict['M'] = 2.59283
        dict['O'] = 00
    if season=="fall":
        dict['A'] = 2.2352
        dict['B'] = 1.78816
        dict['C'] = 0.7120      # Easy Pass forest
        dict['E'] = 1.6412
        dict['F'] = 1.7956
        dict['G'] = 1.0234
        dict['H'] = 0.5210
        dict['K'] = 2.68224
        dict['M'] = 2.59283
        dict['O'] = 00
    if season=="winter":
        dict['A'] = 2.2352
        dict['B'] = 1.78816
        dict['C'] = 2.0120
        dict['E'] = 1.6412
        dict['F'] = 1.7956
        dict['G'] = 1.0234
        dict['H'] = 0.5210
        dict['K'] = 2.68224
        dict['M'] = 2.59283
        dict['P'] = 1.1458      # Ice
        dict['O'] = 00
    if season=="spring":
        dict['A'] = 2.2352
        dict['B'] = 1.78816
        dict['C'] = 2.0120
        dict['E'] = 1.6412
        dict['F'] = 1.7956
        dict['G'] = 1.0234
        dict['H'] = 0.5210
        dict['K'] = 2.68224
        dict['M'] = 2.59283
        dict['Q'] = 0.5210      # Mud-Swamp
        dict['O'] = 00
    return dict

"""
This method calculates gradient difference and makes it steeper for climbing
and easier fo down fall

:param  fromNode    Start node
:param  toNode      Destination node 
"""
def calculate_gradient_difference(fromNode,toNode):
    gradient = fromNode[2]-toNode[2]
    if(gradient>0):
        cost = gradient*1.1
    elif(gradient<0):
        cost=gradient*0.9
    else:
        cost=gradient
    return cost

"""
This method implements A* Search Algorithm to generate graph for drawing path to the destination node

:param  node1                   Start node
:param  node2                   Target node
:param  season_terrain_data     Season's terrain Data
:param  merged_elevation_data   List of neighbours of nodes
:param  speed_dict              Dictionary of speed of terrains
"""

def find_next_neighbour(node1,node2,season_terrain_data,elevation_data,merged_elevation_data,speed_dict):
    pq=[]
    result={}
    targetNode = node2.copy()
    targetNode.append(elevation_data[node2[0]][node2[1]])
    currentNode = merged_elevation_data[node1[0]][node1[1]]
    childNode = node1.copy()
    childNode.append(elevation_data[node1[0]][node1[1]])

    visited={}
    parent={}
    visited[str(childNode)]=0

    while(childNode!=targetNode):
        node = childNode

        for neighbour in currentNode:
            if speed_dict[season_terrain_data[neighbour[0]][neighbour[1]]] != 0:
                nextCost = ((elevation_data[node[0]][node[1]] - elevation_data[neighbour[0]][neighbour[1]]) / speed_dict[season_terrain_data[neighbour[0]][neighbour[1]]])
                movement_cost = visited[str(childNode)] + nextCost

                if str(neighbour) not in visited :
                    visited[str(neighbour)] = movement_cost

                    heuristic_time =  calculate_heuristic_value(neighbour,node,season_terrain_data)

                    totalCost = movement_cost+heuristic_time
                    pq.append((totalCost,neighbour))
                    parent[str(neighbour)]=node

        pq.sort(reverse=True)

        resultNode = pq.pop()

        childNode = resultNode[1]
        currentNode=merged_elevation_data[childNode[0]][childNode[1]]
        distance = abs(childNode[2] - node[2])
        result[str(childNode)]=[parent[str(childNode)],abs(movement_cost),distance]
        # print(resultNode,":",targetNode)

    return result

"""
This method draws path in outout file as derived by a* search

:param  in_file         input file path
:param  out_file        output file path
:param  path_values     path to be traced
:param  path_data       path to mark spots
:param  color           colour of path
"""
def draw_path(in_file,out_file,path_values,path_data,color):
    print("Generating Output Image")
    im = Image.open(in_file)
    img=im.load()

    for i in path_values:
        img[i[0],i[1]]=color
    for j in path_data:
        img[j[0],j[1]]=(255, 0, 0)
    im.save(out_file)
    print(out_file," file saved")
    im.show(out_file)


"""
This method draws ice in image

:param  in_file         input file path
:param  out_file        output file path
:param  path_values     path to be traced
:param  color           colour of path
"""
def draw_ice(in_file,out_file,path_values,color):
    print("Ice Terrain is being generated")
    im = Image.open(in_file)
    img=im.load()

    for i in path_values:
        img[i[0],i[1]]=color
    im.save(out_file)
    print(out_file," file saved")

"""
This method draws mud in image

:param  in_file         input file path
:param  out_file        output file path
:param  path_values     path to be traced
:param  color           colour of path
"""
def draw_mud(in_file,out_file,path_values,color):
    print("Mud Terrain is being generated")
    im = Image.open(in_file)
    img=im.load()

    for i in path_values:
        img[i[0],i[1]]=color
    im.save(out_file)
    print(out_file," file saved")

"""
This method generates path from the graph derived by A* search
:param  node1   start Node
:param  node2   finish node
:param  path    Graph of the connected components
"""
def makePath(node1,node2,path):
    path_way=[]
    start = node1.copy()
    start.append(elevation_data[start[0]][start[1]])
    finish = node2.copy()
    finish.append((elevation_data[finish[0]][finish[1]]))
    sum=0
    distance=0
    path_way.append(finish[0:2])
    if str(finish) in path:
        tempNode=path[str(finish)][0]
        sum+=path[str(finish)][1]
        distance += path[str(finish)][2]


        while (tempNode != start):
            path_way.append(tempNode[0:2])
            sum += path[str(tempNode)][1]
            distance += path[str(tempNode)][2]
            tempNode=path[str(tempNode)][0]

    path_way.append(start[0:2])

    return path_way,sum,distance

"""
This method calculates heuristic valuesi n terms of time from
given node to target node

:param  fromNode                from the node co-ordinates
:param  toNode                  to node co-ordinates
:param  season_terrain_data     seasonal terrain values
"""
def calculate_heuristic_value(fromNode,toNode,season_terrain_data):
    node1=[fromNode[0],fromNode[1]]
    node2=[toNode[0],toNode[1]]
    tempDistance = calculate_euclidian_distance(node1, node2)
    speed = speed_dict[season_terrain_data[fromNode[0]][fromNode[1]]]
    ans = tempDistance / speed
    return ans


"""
This method calculates Manhattan distance between two given nodes

:param  node1   from node
:param  node2   to node
"""
def calculate_euclidian_distance(node1,node2):
    xfactor = 10.29
    yfactor = 7.5
    xDist = ((node1[0]-node2[0]) * xfactor) ** 2
    yDist = ((node1[1]-node2[1]) * yfactor) ** 2
    distance = math.sqrt(xDist+yDist)
    return distance

"""
This is the  main method
"""
if __name__=="__main__":
    print("lab 1")
    rows, cols = (500, 375)

    terrain_file = sys.argv[1]
    elevation_file_name = sys.argv[2]
    path_file = sys.argv[3]
    season = sys.argv[4]
    output_file = sys.argv[5]

    seasons=["summer","winter","spring","fall"]
    season=season.lower()
    if season not in seasons:
        print("Enter valid season")
        sys.exit()

    pathResult=[]
    path_total = 0
    total_distance = 0
    path_records={}

    path_data,file_length=load_path_file(path_file)

    elevation_data = load_elevation_file(elevation_file_name)
    terrain_data=load_terrain_file(terrain_file)

    merged_elevation_data = make_neighbours(elevation_data)
    speed_dict = generate_speed_dictionary(season)
    print("For Season : ",season)
    if season=="summer" or season == "fall":

        for i in range(file_length-1):

            path= find_next_neighbour(path_data[i], path_data[i+1],terrain_data, elevation_data, merged_elevation_data, speed_dict)
            path_way,path_distance,distance = makePath(path_data[i], path_data[i+1], path)
            pathResult.append(path_way)
            path_total +=path_distance
            total_distance += distance
            path_records[str([path_data[i], path_data[i+1]])]=path_distance

        finalPath=[]
        for line in pathResult:
            for pix in line:
                finalPath.append(pix)
        draw_path(terrain_file,output_file,finalPath,path_data,(152, 98, 181))
        print("Total distance of path is : ", distance," meters")
        print("Total Time of path is : ", path_total," seconds")
        print("Time between each point is :")
        for record,time in path_records.items():
            print(record,"\t:\t",time)
    if season == "winter":
        print("winter")
        ice_path,winter_terrain_data=make_ice_path(merged_elevation_data,terrain_data)

        out_file= "ice.png"
        draw_ice(terrain_file, out_file, ice_path, (0, 255, 229))
        for i in range(file_length-1):
            path= find_next_neighbour(path_data[i], path_data[i+1], winter_terrain_data,elevation_data, merged_elevation_data, speed_dict)
            path_way,path_distance,distance = makePath(path_data[i], path_data[i+1], path)
            pathResult.append(path_way)
            path_total += path_distance
            total_distance += distance
            path_records[str([path_data[i], path_data[i + 1]])] = path_distance

        finalPath=[]
        for line in pathResult:
            for pix in line:
                finalPath.append(pix)
        draw_path(out_file,output_file,finalPath,path_data,(152, 98, 181))
        print("Total distance of path is : ", distance, " meters")
        print("Total Time of path is : ", path_total, " seconds")
        print("Time between each point is :")
        for record, time in path_records.items():
            print(record, "\t:\t", time)

    if season == "spring":
        print("spring")

        mud_path,spring_terrain_data = make_mud_path(elevation_data,merged_elevation_data,terrain_data)
        out_file = "mud.png"
        draw_mud(terrain_file, out_file, mud_path, (163, 121, 121))
        for i in range(file_length-1):
            path= find_next_neighbour(path_data[i], path_data[i+1], spring_terrain_data,elevation_data, merged_elevation_data, speed_dict)
            path_way,path_distance,distance = makePath(path_data[i], path_data[i+1], path)
            pathResult.append(path_way)
            path_total += path_distance
            total_distance += distance
            path_records[str([path_data[i], path_data[i + 1]])] = path_distance

        finalPath=[]
        for line in pathResult:
            for pix in line:
                finalPath.append(pix)
        draw_path(out_file,output_file,finalPath,path_data,(152, 98, 181))
        print("Total distance of path is : ", distance, " meters")
        print("Total Time of path is : ", path_total, " seconds")
        print("Time between each point is :")
        for record, time in path_records.items():
            print(record, "\t:\t", time)
