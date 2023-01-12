package com.momentum4999.vision;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class VisionMain {
    private static final String WINDOW_ID = "nano-vision";

    public static void main(String[] args) throws IOException {
        String settingsPath = "settings.txt";
        String hostname = "localhost";
        if (args.length >= 1) {
            settingsPath = args[0];
        }
        if (args.length >= 2) {
            hostname = args[1];
        }

        Properties settings = new Properties();
        try (InputStream settingsFile = new FileInputStream(settingsPath)) {
            settings.load(settingsFile);
        }
        boolean gui = Boolean.parseBoolean(settings.getProperty("gui", "false"));

        VisionNetworkTable table = new VisionNetworkTable(hostname);
        AprilTagVision aprilTag = new AprilTagVision(table, settings);

        if (gui) {
            HighGui.namedWindow(WINDOW_ID);
        }
        VideoCapture video = new VideoCapture(0);
        Mat imageBuffer = new Mat();
        while (true) {
            video.read(imageBuffer);

            aprilTag.update(imageBuffer);

            if (gui) {
                HighGui.imshow(WINDOW_ID, imageBuffer);
                if (HighGui.waitKey(10) == java.awt.event.KeyEvent.VK_Q) {
                    break;
                }
            }
        }

        HighGui.destroyAllWindows();
        aprilTag.close();
        System.exit(0);
    }

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
}
