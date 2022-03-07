import xml.etree.ElementTree as ET

def get_frames(manga):
    """
    function that extracts the frames (text region) from all pages of
    a book
    args :
        - manga (str) : path to xml file containing annotation
    retr : dictionnary of regions
    """
    frames = {}
    tree = ET.parse(manga)
    root = tree.getroot()
    root = root[1] #root contains all pages
    for page in root.findall('page'):
        frames[int(page.get("index"))] = [[frm.get("xmin"),
                                      frm.get("ymin"),
                                      frm.get("xmax"),
                                      frm.get("ymax")]
                                      for frm in page.findall('text')]
    return frames