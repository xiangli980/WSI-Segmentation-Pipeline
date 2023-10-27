import numpy as np
import cv2

def get_contour_points(mask, downsample,value, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    pointsList = []
    min_size = [30,30,30,30,30,30]
    for j in np.array(range(len(maskPoints))):
        if len(maskPoints[j])>2:
            if cv2.contourArea(maskPoints[j]) > min_size[value-1]:
                pointList = []
                for i in np.array(range(0,len(maskPoints[j]),4)):
                    point = {'X': (maskPoints[j][i][0][0] * downsample) + offset['X'], 'Y': (maskPoints[j][i][0][1] * downsample) + offset['Y']}
                    pointList.append(point)
                pointsList.append(pointList)
    return pointsList

def converter(geojson_list, names, colorList=None, alpha=0.4):
    if colorList == None:
        colorList = ["rgb(0, 255, 128)", "rgb(0, 255, 255)", "rgb(255, 255, 0)", "rgb(255, 128, 0)", "rgb(0, 128, 255)",
                     "rgb(0, 0, 255)", "rgb(0, 102, 0)", "rgb(153, 0, 0)", "rgb(0, 153, 0)", "rgb(102, 0, 204)",
                     "rgb(76, 216, 23)", "rgb(102, 51, 0)", "rgb(128, 128, 128)", "rgb(0, 153, 153)", "rgb(0, 0, 0)"]
    data = []

    for n, child in enumerate([geojson_list]):
        dataDict = dict()
        name = names[n]
        #_ = os.system("printf 'Building JSON layer: [{}]\n'".format(name))
        element = []
        
        for i in child:
            eleDict = dict()
            eleDict["closed"] = True

            lineColor = colorList[n % len(colorList)]
            eleDict["lineColor"] = lineColor

            fillColor = lineColor[:3]+'a'+lineColor[3:-1] + f', {alpha})'
            eleDict["fillColor"] = fillColor

            eleDict["lineWidth"] = 2
            points = []
            #ver = i.find('Vertices')
            Verts = i
            if len(Verts) <= 1:
                continue # skip if only 1 vertex points
            for j in Verts:
                eachPoint = []
                eachPoint.append(float(j['X']*16))
                eachPoint.append(float(j['Y']*16))
                eachPoint.append(float(0))

                points.append(eachPoint)
            eleDict["points"] = points
            eleDict["type"] = "polyline"
            element.append(eleDict)
        dataDict["elements"] = element
        dataDict["name"] = name
        data.append(dataDict)

    return data