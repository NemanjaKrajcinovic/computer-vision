import numpy as np
from cv2 import putText, circle, LINE_AA, FONT_HERSHEY_PLAIN
# Keras
from Estimator.Age_and_Gender_Estimation_Keras import AgeAndGenderEstimator
'''
# CNN
from Estimator.Age_and_Gender_Estimation import AgeAndGenderEstimator
'''


class TrackableObject:
    age_and_gender_estimator = AgeAndGenderEstimator()
    next_person_ID = 0
    '''
    # CNN
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']
    '''

    def __init__(self, persons_downoid):
        # Store the object ID, current centroid
        self.persons_ID = TrackableObject.next_person_ID
        TrackableObject.next_person_ID += 1
        self.persons_downoid = persons_downoid
        self.persons_transformed_downoid = (0, 0)
        self.persons_disappearing = 0

        self.persons_ages = 0
        self.persons_gender = 0
        self.age_and_gender_averager = 0
        self.already_estimated = False

        self.persons_trajectory = list()
        self.trajectory_input_counter = 0
        self.color = np.random.choice(range(256), size=3)
        self.color = (int(self.color[0]), int(self.color[1]), int(self.color[2]))

    def update_downoid(self, new_downoid):
        if self.trajectory_input_counter == 10:
            self.persons_trajectory.append(self.persons_transformed_downoid)
            self.trajectory_input_counter = 0
        self.persons_downoid = new_downoid
        self.trajectory_input_counter += 1

    def estimate_age_gender(self, persons_image):
        if self.already_estimated is False:
            # Start to estimate person's age and gender.
            # At first, go through 10 frames and do the estimation on the every single frame.
            # After that, average the result and assign real values.
            if self.age_and_gender_averager < 10:
                temporary_age = self.persons_ages
                temporary_gender = self.persons_gender

                self.persons_ages, self.persons_gender = TrackableObject.age_and_gender_estimator.estimate_age_and_gender(persons_image)

                if self.persons_ages is not None and self.persons_gender is not None:
                    self.persons_ages += temporary_age
                    self.persons_gender += temporary_gender
                    self.age_and_gender_averager += 1

                if self.persons_ages is None or self.persons_gender is None:
                    self.persons_ages = temporary_age
                    self.persons_gender = temporary_gender

            if self.age_and_gender_averager == 10:
                final_age = int(self.persons_ages / 10)
                # Keras
                final_gender = self.persons_gender / 10
                self.persons_ages = final_age
                self.persons_gender = "Female" if final_gender > 0.5 else "Male"
                '''
                # CNN
                final_gender = int(self.persons_gender / 10)
                self.persons_ages = self.age_list[final_age - 1]
                self.persons_gender = self.gender_list[final_gender - 1]
                '''
                self.already_estimated = True

    def draw_on_frame(self, frame):
        # Draw both the ID and estimated age and gender of the person.
        text = "ID {}".format(self.persons_ID)
        putText(frame, text, (self.persons_downoid[0] - 30, self.persons_downoid[1] - 35), FONT_HERSHEY_PLAIN, 1, self.color, 1)
        circle(frame, (self.persons_downoid[0], self.persons_downoid[1]), 4, self.color, -1)
        if type(self.persons_ages) == int and self.persons_ages != 0:
            putText(frame, "Ages: " + str(self.persons_ages), (self.persons_downoid[0] - 30, self.persons_downoid[1] - 20), FONT_HERSHEY_PLAIN, 1, self.color, 1, LINE_AA)
        if type(self.persons_gender) == str:
            putText(frame, "Gender: " + str(self.persons_gender), (self.persons_downoid[0] - 30, self.persons_downoid[1] - 5), FONT_HERSHEY_PLAIN, 1, self.color, 1, LINE_AA)
