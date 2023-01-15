package com.momentum4999.vision;

import edu.wpi.first.cscore.CvSink;
import edu.wpi.first.cscore.CvSource;
import edu.wpi.first.cscore.MjpegServer;
import edu.wpi.first.cscore.UsbCamera;
import edu.wpi.first.cscore.VideoMode;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.Properties;

public class VisionMain {
    private static final String VIDEO_WINDOW = "nano-vision-video";
    private static final String FIELD_WINDOW = "nano-vision-field";

    public static void main(String[] args) throws IOException {
        String settingsPath = "settings.txt";
        if (args.length >= 1) {
            settingsPath = args[0];
        }

        Properties settings = new Properties();
        try (InputStream settingsFile = new FileInputStream(settingsPath)) {
            settings.load(settingsFile);
            System.out.println("Loaded settings from file " + new File(settingsPath).getAbsolutePath());
        }
        boolean gui = Boolean.parseBoolean(settings.getProperty("gui", "false"));
        boolean drawField = Boolean.parseBoolean(settings.getProperty("gui.field", "true"));

        VisionNetworkTable table = new VisionNetworkTable(settings.getProperty("nt.server", "localhost"));
        AprilTagVision aprilTag = new AprilTagVision(table, settings);

        if (gui) {
            System.out.println("GUI Enabled");
            HighGui.namedWindow(VIDEO_WINDOW);
            if (drawField) {
                HighGui.namedWindow(FIELD_WINDOW);
            }
        }
        int vWidth = Integer.parseInt(settings.getProperty("video.width", "640"));
        int vHeight = Integer.parseInt(settings.getProperty("video.height", "480"));
        int vFramerate = Integer.parseInt(settings.getProperty("video.fps", "30"));
        int vSrvPort = Integer.parseInt(settings.getProperty("video.server_port", "1300"));
        float vOutScale = Float.parseFloat(settings.getProperty("video.output_scale", "0.5"));

        CvSink videoIn = new CvSink("cv_nvcam");
        CvSource videoOut = new CvSource("nvcam", VideoMode.PixelFormat.kMJPEG, vWidth, vHeight, vFramerate);

        UsbCamera camera = new UsbCamera("nvcam", 0);
        videoIn.setSource(camera);

        MjpegServer videoServer = new MjpegServer("nvcam_server", vSrvPort);
        videoServer.setSource(videoOut);
        String hostname = "localhost";
        try {
            hostname = InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException ignored) {}
        System.out.format("Camera server started at %s:%s\n", hostname, vSrvPort);

        final Mat fieldImage;
        try (InputStream fieldImageFile = VisionMain.class.getResourceAsStream("/field.png")) {
            if (fieldImageFile != null) {
                fieldImage = Imgcodecs.imdecode(new MatOfByte(fieldImageFile.readAllBytes()), Imgcodecs.IMREAD_COLOR);
            } else {
                throw new RuntimeException("Field image resource unavailable");
            }
        }

        Mat videoBuffer = new Mat();
        Mat fieldBuffer = new Mat();
        Mat outputBuffer = new Mat();
        boolean error = false;
        while (true) {
            long frameStatus = videoIn.grabFrame(videoBuffer);
            if (frameStatus > 0) {
                error = false;
                aprilTag.updateVideo(videoBuffer);

                if (drawField) {
                    fieldImage.copyTo(fieldBuffer);
                    aprilTag.updateField(fieldBuffer);
                }

                if (gui) {
                    HighGui.imshow(VIDEO_WINDOW, videoBuffer);
                    if (drawField) {
                        HighGui.imshow(FIELD_WINDOW, fieldBuffer);
                    }
                    if (HighGui.waitKey(10) == java.awt.event.KeyEvent.VK_Q) {
                        break;
                    }
                }

                Imgproc.resize(videoBuffer, outputBuffer, new Size((int) (vWidth * vOutScale), (int) (vHeight * vOutScale)));
                videoOut.putFrame(outputBuffer);
            } else if (!error) {
                error = true;
                System.err.println("Error grabbing video feed: grabFrame() returned status code 0");
            }
        }

        System.out.println("Ending...");
        HighGui.destroyAllWindows();
        aprilTag.close();
        System.exit(0);
    }

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
}
