import cv2
import yolo_images_2
import tkinter as tk

# Code to add widgets will go here...
class Window_1st:
    def __init__(self, master):
        self.master = master
        self.master.geometry('656x490')
        self.master.title("")

        self.master.resizable(False, False)
        self.filename = tk.PhotoImage(
            file = "C:/Users/DELL/PycharmProjects"
                   +"/deep_learning_Custom_model/gun8.png"
        )
        self.background_label = tk.Label(
            self.master, image=self.filename
        )
        self.background_label.place(
            x=0, y=0, relwidth=1, relheight=1
        )


        self.HelloButton = tk.Button(
            self.master,
            text='next',
            width=25,
            command=self.new_window,
        )
        # self.HelloButton.pack()
        self.HelloButton.grid(
            row=3, column=2, padx=230,pady=455
        )
        self.HelloButton.configure(
            bg='#45ada8', fg='white'
        )
        #
    def close_windows(self):
        self.master.destroy()
        self.new_window

    def new_window(self):
        self.master.destroy()  # close the current window
        self.master = tk.Tk()  # create another Tk instance
        self.app = Window_2nd(self.master)  # create Window_2nd window
        self.master.mainloop()

class Window_2nd:
    def __init__(self, master):
        self.master = master
        self.master.geometry('656x490')
        self.master.title("")
        self.master.resizable(False, False)

        self.filename = tk.PhotoImage(
            file = "C:/Users/DELL/PycharmProjects/"+
                   "deep_learning_Custom_model/gun66.png"
        )
        self.background_label = tk.Label(
            self.master, image=self.filename
        )
        self.background_label.place(
            x=0, y=0, relwidth=1, relheight=1
        )


        self.quitButton = tk.Button(
            self.master,
            text = 'Start',
            width = 25,
            command = self.start_canvas
        )


        self.quitButton.grid(
            row=3, column=2, padx=230, pady=455
        )
        self.quitButton.configure(bg='#45ada8',fg='white')
        self.master.configure(background='red')

    def close_windows(self):
        self.master.destroy()


    def start_canvas(self):
        global lable_image
        obj_detect = yolo_images_2.Detection()

        for img_path in obj_detect.images_path:
            images = obj_detect.detect_images(img_path)
            # if(obj_detect.IsDetected):
            #     winsound.Beep(1000, 1000)  # Beep at 1000 Hz for 100 ms

            # Rearrang the color channel
            # b, g, r = cv2.split(images)
            # img = cv2.merge((r, g, b))

            # Convert the Image object into a TkPhoto object
            # im = Image.fromarray(img)
            # imgtk = ImageTk.PhotoImage(image=img)

            obj_detect.IsDetected = False
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break




def main():
    root = tk.Tk()
    app = Window_1st(root,)

    root.mainloop()

if __name__ == '__main__':
    main()