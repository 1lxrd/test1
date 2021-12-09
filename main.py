import argparse
import Classes
import cv2
import os
import string
import json


def form_destination(path):
    slash = path.rfind('\\')
    destination = ""

    if slash == -1:
        destination = "out_" + path

    else:
        for i in range(0, slash + 1):
            destination += path[i]

        destination += 'out_'

        for i in range(slash + 1, len(path)):
            destination += path[i]

    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True)
    parser.add_argument("-d", "--destination", default=-1)
    parser.add_argument("-r", "--report", default='report.json')
    parser.add_argument("--grayscale", action='store_true')
    parser.add_argument("--outline", action='store_true')
    parser.add_argument("-t", "--templates", default=-1)

    args = parser.parse_args()

    report_dict = {}
    files_list = []

    try:

        for filename in os.listdir(args.source):
            if filename.endswith(".jpg"):
                path_to_image = os.path.join(args.source, filename)
                files_list.append(filename)
                image = Classes.NewImage(path_to_image)

                if args.destination == -1:
                    save_destination = form_destination(path_to_image)
                    save_destination = str(save_destination)

                else:
                    save_destination = args.destination

                if args.outline:
                    image.outline(save_destination)
                    transform = {'transformations': 'outline'}
                    report_dict.update(transform)
                    image.save(save_destination)

                elif args.grayscale:
                    transform = {'transformations': 'grayscale'}
                    report_dict.update(transform)
                    image.save(save_destination)

                elif args.templates != -1:
                    for templatename in os.listdir(args.templates):
                        search_template = ''
                        for i in range(0, len(filename)-4):
                            search_template += filename[i]
                        search_template += '_template.png'

                        if templatename == search_template:
                            image.outline(save_destination)
                            transform = {'transformations': 'outline'}
                            report_dict.update(transform)

                            path_to_template = os.path.join(args.templates, templatename)

                            temp = Classes.TemplateFinder(path_to_template)

                            temp.template.cv_image = cv2.cvtColor(cv2.imread(path_to_template), cv2.COLOR_BGR2GRAY)
                            grayscale = temp.template.cv_image

                            ret, thresh = cv2.threshold(grayscale, 127, 255, 0)

                            con, hir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                            cv_image = cv2.cvtColor(temp.template.cv_image, cv2.COLOR_GRAY2RGB)
                            cv2.drawContours(temp.template.cv_image, con, -1, (0, 255, 0), 2)

                            a, b, c = temp.find(image)
                            cv2.rectangle(image, a, b, (0, 0, 255), thikness=3)
                            image.save(save_destination)

        files = {'file_list': files_list}
        report_dict.update(files)

    except Exception as e:

        exceptions = {'exc': str(e)}
        report_dict.update(exceptions)

    with open(args.report, 'w') as fp:
        json.dump(report_dict, fp)
