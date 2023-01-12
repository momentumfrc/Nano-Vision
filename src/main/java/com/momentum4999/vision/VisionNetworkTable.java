package com.momentum4999.vision;

import edu.wpi.first.networktables.DoublePublisher;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableInstance;

public class VisionNetworkTable {
    private final DoublePublisher cameraX;
    private final DoublePublisher cameraY;
    private final DoublePublisher cameraYaw;

    public VisionNetworkTable(String robotHostname) {
        NetworkTableInstance inst = NetworkTableInstance.getDefault();
        NetworkTable vision = inst.getTable("nano_vision");
        this.cameraX = vision.getDoubleTopic("camera_x").publish();
        this.cameraY = vision.getDoubleTopic("camera_y").publish();
        this.cameraYaw = vision.getDoubleTopic("camera_yaw").publish();

        inst.startClient4("Nano Vision");
        inst.setServer(robotHostname);
    }

    public void updatePose(double x, double y, double yaw) {
        this.cameraX.accept(x);
        this.cameraY.accept(y);
        this.cameraYaw.accept(yaw);
    }
}
