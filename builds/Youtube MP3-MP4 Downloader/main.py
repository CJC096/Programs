import os
import yt_dlp
import tkinter as tk
from tkinter import messagebox

# Get the user's Downloads directory dynamically
DOWNLOAD_DIR = os.path.join(os.path.expanduser('~'), 'Downloads')


def download_video(url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Best video and audio
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),  # Save to Downloads folder
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    messagebox.showinfo("Success", "Download completed!")


def download_audio_only(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',  # Extract audio using FFmpeg
                'preferredcodec': 'mp3',     # Save as MP3
                'preferredquality': '192',   # Quality (bitrate)
            }
        ],
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),  # Save to Downloads folder
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    messagebox.showinfo("Success", "Download completed!")


def download_video_only(url):
    ydl_opts = {
        'format': 'bestvideo',  # Only video
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),  # Save to Downloads folder
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    messagebox.showinfo("Success", "Download completed!")


def download_playlist(url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Best video and audio
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(playlist_title)s/%(playlist_index)s - %(title)s.%(ext)s'),
        'noplaylist': False,  # Enable playlist download
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    messagebox.showinfo("Success", "Download completed!")


def download_channel(url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(uploader)s/%(title)s.%(ext)s'),  # Organize by uploader
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    messagebox.showinfo("Success", "Download completed!")


def start_download():
    url = url_entry.get()
    selected_option = option_var.get()

    if not url:
        messagebox.showwarning("Input Error", "Please enter a valid URL.")
        return

    if selected_option == "Playlist":
        download_playlist(url)
    elif selected_option == "Video":
        download_video(url)
    elif selected_option == "Channel":
        download_channel(url)
    elif selected_option == "Audio":
        download_audio_only(url)
    elif selected_option == "Image":
        download_video_only(url)
    else:
        messagebox.showwarning("Invalid Selection", "Please select a valid option.")


# Set up the GUI  window
root = tk.Tk()
root.title("YouTube Downloader")

# Dropdown for selecting the type of download
option_var = tk.StringVar(root)
option_var.set("Video")  # Default selection
options = ["Playlist", "Video", "Channel", "Audio", "Image"]
dropdown = tk.OptionMenu(root, option_var, *options)
dropdown.grid(row=0, column=0, padx=10, pady=10)

# Label for the URL input
url_label = tk.Label(root, text="Enter YouTube URL:")
url_label.grid(row=1, column=0, padx=10, pady=5)

# Textbox for the user to input the URL
url_entry = tk.Entry(root, width=40)
url_entry.grid(row=1, column=1, padx=10, pady=5)

# Download button
download_button = tk.Button(root, text="Start Download", command=start_download)
download_button.grid(row=2, column=0, columnspan=2, pady=20)

# Start the Tkinter event loop
root.mainloop()
