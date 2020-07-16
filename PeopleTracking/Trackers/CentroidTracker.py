#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################################
#
# Created by Natasa Avramovic
# email: avramovicnatasa97@gmail.com
#
#
# Persons tracker based on bounding box of every detected person.
# Calculations are made within centroids of objects.
#
# changes:
# - 14.5.2020. - Nemanja Krajcinovic
# - Improvements in tracking adding maximum distance betwen two frames for one object.
# - 21.5.2020. - Natasa Avramovic
# - Added function distance() from scipy for determining distance between two centroids.
#
############################################################################
from Trackers.TrackableObject import TrackableObject
from scipy.spatial import distance
import numpy as np


class CentroidTracker:
    '''
    Class for tracking multiple persons.
    '''
    def __init__(self, set_max_disappear=25, set_max_distance=75):
        '''
        Initialisation of object for People tracking.
        '''
        # Number of maximum consecutive frames a given object is allowed to be marked as "disappeared" until deregistration.
        self.max_disappear = set_max_disappear

        # Store the maximum distance between centroids to associate an object.
        self.max_distance = set_max_distance

    def update(self, rectagles, persons_images, persons):
        '''
        Tracking people using "persons" dictionary. Dictionary is updated every time when new frame has been processed.
        At first, there is a check if none of persons are found on the current frame to deregister every person which has not been present enough.
        Second, for every detected person on the current frame its centroid is calculated.
        Then, there is a check if none of persons are tracked to start tracking every found person.
        Otherwise, euclidean distance is calculated for every pair of new centroid and tracekd person.
        After that, every distance is examined and if it is smaller then the maximum distance which is set, new centroid is connected to tracked person.
        New centroids which are not connected to tracked persons are finally registered as newly tracked persons.
        '''
        # (1)
        # Check to see if the list of input bounding box rectangles is empty.
        if len(rectagles) == 0:
            # If the list is empty,
            # loop over any existing tracked persons and mark them as disappeared because they do not present in the current frame.
            for i in range(0, len(persons)):
                persons[i].persons_disappearing += 1
                # If a maximum number of consecutive frames has been reached for a given object which has been marked as missing, deregister it.
                if persons[i].persons_disappearing > self.max_disappear:
                    del persons[i]
            # Return remaining persons which are still tracked.
            return persons

        # (2)
        # Initialise an array of input centroids for the current frame.
        input_downoids = np.zeros((len(rectagles), 2), dtype="int32")

        # Loop over the input bounding box rectangles, calculate their centroids and store them in array.
        for (i, (start_x, _, end_x, end_y)) in enumerate(rectagles):
            c_x = int((start_x + end_x) / 2.0)
            input_downoids[i] = (c_x, end_y)

        # If none of persons are currently tracked, take the input centroids and register each of them.
        if len(persons) == 0:
            for i in range(0, len(input_downoids)):
                persons.append(TrackableObject(input_downoids[i]))
        else:
            # Otherwise, there are persons which are currently tracked, so try to match the input centroids to existing object centroids.
            persons_IDs = list()
            persons_downoids = list()

            for i in range(0, len(persons)):
                persons_IDs.append(persons[i].persons_ID)
                persons_downoids.append(persons[i].persons_downoid)

            # Compute the distance between each pair of object centroids and input centroids.
            distance_between_downoids = distance.cdist(np.array(persons_downoids), input_downoids)

            # In order to perform this matching:
            # 1. Find the smallest value in each row,
            # 2. Sort the row indexes based on their minimum values.
            rows = distance_between_downoids.min(axis=1).argsort()

            # Next, perform a similar process on the columns:
            # Find the smallest value in each column and then sort using the previously computed row index list.
            columns = distance_between_downoids.argmin(axis=1)[rows]

            # In order to determine is it needed to update, register, or deregister an object,
            # keep track which of the row and column indexes we have already examined.
            used_rows = set()
            used_columns = set()

            # Next, loop over the combination of the (row, column) and examine every measured distance.
            for (row, column) in zip(rows, columns):
                # If some distance is already examined, either the row or column value, ignore it.
                if row in used_rows or column in used_columns:
                    continue

                # If the distance between centroids is greater than the maximum distance,
                # do not associate these two centroids to the same object.
                if distance_between_downoids[row, column] > self.max_distance:
                    continue

                # Otherwise, the person is already tracked, so:
                # Grab the object ID for the current row and set its new centroid and reset the disappearing counter.
                person_ID = persons_IDs[row]
                for i in range(0, len(persons)):
                    if persons[i].persons_ID == person_ID:
                        persons[i].update_downoid(input_downoids[column])
                        persons[i].persons_disappearing = 0
                        persons[i].estimate_age_gender(persons_images[column])
                        break

                # Indicate that each of the row and column has been examined.
                used_rows.add(row)
                used_columns.add(column)

            # Next, find both the row and column indexes which still have not been examined.
            unused_rows = set(range(0, distance_between_downoids.shape[0])).difference(used_rows)
            unused_columns = set(range(0, distance_between_downoids.shape[1])).difference(used_columns)

            # If the number of object centroids is equal or greater than the number of input centroids,
            # check if some of these persons have potentially disappeared,
            # or if they have not disappeared increment their counter for "disappearing".
            if distance_between_downoids.shape[0] >= distance_between_downoids.shape[1]:
                for row in unused_rows:
                    person_ID = persons_IDs[row]
                    for i in range(0, len(persons)):
                        if persons[i].persons_ID == person_ID:
                            persons[i].persons_disappearing += 1

                            if persons[i].persons_disappearing > self.max_disappear:
                                del persons[i]

                            break

            # Otherwise, register each new input centroid as a new trackable object.
            else:
                for column in unused_columns:
                    persons.append(TrackableObject(input_downoids[column]))

        return persons
