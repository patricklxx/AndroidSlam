#include "core/App.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <sstream>

#include "utils/Log.h"
#include "utils/AssetManager.h"

#include "render/ImageTexture.h"
#include "render/Shader.h"
#include "render/Plane2D.h"

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

int CHECKERBOARD[2]{6,9};

namespace android_slam
{

    android_app* g_state;

    App::App(android_app* state) noexcept
    {
        static std::atomic_bool is_init = false;
        assert((!is_init) && "App should only have one instance.");

        g_state = state;
        state->userData = this;
        state->onAppCmd = App::onCmd;
        state->onInputEvent = App::onInput;

        SensorTexture::registerFunctions();
    }

    void App::run()
    {
        m_timer.mark();
        while (m_running)
        {
            // Handle events.
            {
                int32_t event;
                android_poll_source* source;

                if(ALooper_pollAll((m_window != nullptr ? 1 : 0), nullptr, &event, reinterpret_cast<void**>(&source)) >= 0)
                {
                    if(source)
                    {
                        source->process(g_state, source);
                    }
                }
            }

            if(!m_active)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }

            float dt = m_timer.mark();
            if(dt < k_min_frame_time_second)
            {
                std::this_thread::sleep_for(Timer::Duration(k_min_frame_time_second - dt));
                dt = k_min_frame_time_second;
            }

            update(dt);
        }
    }

    void App::init()
    {
        AssetManager::set(g_state->activity->assetManager);

        m_window = std::make_unique<Window>(
                g_state->window,
                ANativeWindow_getWidth(g_state->window),
                ANativeWindow_getHeight(g_state->window),
                App::k_app_name
        );


        // Camera image converter.
        m_image_pool = std::make_unique<ImagePool>(
            k_sensor_camera_width,
            k_sensor_camera_height,
            "shader/yuv2rgb.vert",
            "shader/yuv2rgb.frag"
        );


        m_active = true;
    }

    void App::exit()
    {
        m_active = false;


        m_image_pool.reset(nullptr);


        m_window.reset(nullptr);
    }

    void App::update(float dt)
    {
        // Clear buffers.
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        // Use GPU shader to trans YUV to RGB.
        {
            std::vector<uint8_t> img = m_image_pool->getImage();
            //vector img to cv::Mat
            cv::Mat cv_img(k_sensor_camera_height,k_sensor_camera_width, CV_8UC3);
            memcpy(cv_img.data, img.data(), sizeof(uint8_t) * img.size());

            //水平翻转
            cv::Mat hv_flip;
            cv::flip(cv_img, hv_flip, 0);

            //旋转45度
            {
                cv::Mat M;
                cv::Mat rot_dst;
                M = cv::getRotationMatrix2D(cv::Point(k_sensor_camera_width/2,k_sensor_camera_height/2),45,1);
                cv::warpAffine(cv_img,rot_dst,M,cv_img.size());
            }

            //仿射变换
            cv::Mat warp_dst;
            cv::Mat warp_mat(2, 3, CV_32FC1);
            {
                cv::Point2f srcTri[3];
                cv::Point2f dstTri[3];

                // 设置源图像和目标图像上的三组点以计算仿射变换
                srcTri[0] = cv::Point2f(0, 0);
                srcTri[1] = cv::Point2f(cv_img.cols - 1, 0);
                srcTri[2] = cv::Point2f(0, cv_img.rows - 1);
                for (size_t i = 0; i < 3; i++){
                    circle(cv_img, srcTri[i], 2, cv::Scalar(0, 0, 255), 5, 8);
                }

                dstTri[0] = cv::Point2f(cv_img.cols * 0.0, cv_img.rows * 0.13);
                dstTri[1] = cv::Point2f(cv_img.cols * 0.95, cv_img.rows * 0.15);
                dstTri[2] = cv::Point2f(cv_img.cols * 0.15, cv_img.rows * 0.9);

                warp_mat = getAffineTransform(srcTri, dstTri);
                warpAffine(cv_img, warp_dst, warp_mat, warp_dst.size());
            }

            //标定
            {
                std::vector<cv::Point3f> obj;
                std::vector<cv::Point2f> corner_pts;
                bool success;
                //定义世界坐标obj
                //obj.clear();
                for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
                {
                    for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
                    {
                        obj.push_back(cv::Point3f(j, i, 0));
                    }
                }
                cv::Mat gray;
                cv::cvtColor(cv_img,gray,cv::COLOR_BGR2GRAY);
                success=cv::findChessboardCorners(gray,cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]),
                                                  corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
                if(success)
                {
                    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::Type::MAX_ITER, 30, 0.001);

                    // 为给定的二维点细化像素坐标
                    cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

                    // 在棋盘上显示检测到的角点
                    cv::drawChessboardCorners(cv_img, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

                    objpoints.push_back(obj);
                    imgpoints.push_back(corner_pts);
                }

                DEBUG_INFO("%lld", imgpoints.size());
                DEBUG_INFO("%lld", objpoints.size());
                if(imgpoints.size()>25)
                {
                    cv::Mat cameraMatrix, distCoeffs, R, T;
                    double rms = cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T,0);
                    objpoints.clear();
                    imgpoints.clear();

                    DEBUG_INFO("RMS == %.3f", rms);
                    DEBUG_INFO(" Matrix_rows = %d", cameraMatrix.rows);
                    DEBUG_INFO(" Matrix_cols = %d", cameraMatrix.cols);
                    for(int i = 0; i < 3; i++)
                    {
                        for(int j = 0; j < 3; j++)
                        {
                            DEBUG_INFO("%.3f ", cameraMatrix.at<float>(i, j));
                        }
                        DEBUG_INFO("\n");
                    }

                    //std::ostringstream oss;
                    //for(int i = 0; i < 3; ++i)
                    //{
                    //    for(int j = 0; j < 3; ++j)
                    //    {
                    //        oss << cameraMatrix.at<float>(i, j) << ' ';
                    //    }
                    //    oss << '\n';
                    //}

                    //auto str = oss.str();
                    //DEBUG_INFO("Matrix = %s", str.c_str());
                }
            }


            //cv::Mat to vector new_img
            std::vector<uint8_t> new_img = cv_img.reshape(1,1);

            glViewport(0, 0, 1280, 960);

            Shader debug_shader("shader/yuv2rgb.vert", "shader/debug_texture.frag");
            Plane2D debug_plane;
            ImageTexture debug_texture(k_sensor_camera_width, k_sensor_camera_height, new_img);

            debug_plane.bind();
            debug_shader.bind();

            glActiveTexture(GL_TEXTURE0);
            debug_shader.setInt("screen_shot", 0);
            debug_texture.bind();

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

            debug_texture.unbind();
            debug_shader.unbind();
            debug_plane.unbind();
        }

        m_window->swapBuffers();

        DEBUG_INFO("[Android Slam App Info] Current FPS: %.3f frame per second.", dt);
    }

    void App::onCmd(android_app *app, int32_t cmd)
    {
        App& instance = *static_cast<App*>(app->userData);

        /*
         * In android, the event is trigger by the following order:
         * Create -> Start -> Resume -> INIT WINDOW
         * ...
         * TERM WINDOW -> Stop -> Destroy
         */

        switch (cmd)
        {
        case APP_CMD_INIT_WINDOW:
        {
            instance.init();

            DEBUG_INFO("App window initialized.");

            break;
        }
        case APP_CMD_TERM_WINDOW:
        {
            instance.exit();

            DEBUG_INFO("App window terminated.");

            break;
        }
        case APP_CMD_WINDOW_RESIZED:
        {
            int32_t new_width = ANativeWindow_getWidth(app->window);
            int32_t new_height = ANativeWindow_getHeight(app->window);

            if(new_width != instance.m_window->getWidth() || new_height != instance.m_window->getHeight())
            {
                instance.m_window->resize(new_width, new_height);
            }

            DEBUG_INFO("App window resized.");

            break;
        }
        case APP_CMD_GAINED_FOCUS:
        {
            instance.m_active = true;

            DEBUG_INFO("App gained focus.");

            break;
        }
        case APP_CMD_LOST_FOCUS:
        {
            instance.m_active = false;

            DEBUG_INFO("App lost focus.");

            break;
        }
        default:
        {}
        }
    }

    int32_t App::onInput(android_app* app, AInputEvent* ie)
    {
        App& instance = *static_cast<App*>(app->userData);

        int32_t event_type = AInputEvent_getType(ie);
        switch (event_type)
        {
        case AINPUT_EVENT_TYPE_KEY:
        case AINPUT_EVENT_TYPE_FOCUS:
        {
            break;
        }
        case AINPUT_EVENT_TYPE_MOTION:
        {
            int32_t action = AMotionEvent_getAction(ie);
            int32_t ptr_idx = (action & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
            action &= AMOTION_EVENT_ACTION_MASK;

            int32_t tool_type = AMotionEvent_getToolType(ie, ptr_idx);

            const float x_value = AMotionEvent_getX(ie, ptr_idx);
            const float y_value = AMotionEvent_getY(ie, ptr_idx);

            switch (action)
            {
            case AMOTION_EVENT_ACTION_DOWN:
            {
                if (tool_type == AMOTION_EVENT_TOOL_TYPE_FINGER ||
                    tool_type == AMOTION_EVENT_TOOL_TYPE_UNKNOWN)
                {
                    instance.onMotionDown(x_value, y_value);
                }
                break;
            }
            case AMOTION_EVENT_ACTION_UP:
            {
                if (tool_type == AMOTION_EVENT_TOOL_TYPE_FINGER ||
                    tool_type == AMOTION_EVENT_TOOL_TYPE_UNKNOWN)
                {
                    instance.onMotionUp(x_value, y_value);
                }
                break;
            }
            case AMOTION_EVENT_ACTION_MOVE:
            {
                instance.onMotionMove(x_value, y_value);
                break;
            }
            case AMOTION_EVENT_ACTION_CANCEL:
            {
                instance.onMotionCancel(x_value, y_value);
                break;
            }
            default:
            {
                break;
            }
            }
        }
        default:
        {
            break;
        }
        }

        if(event_type == AINPUT_EVENT_TYPE_MOTION)
        {
            return 1;
        }
        return 0;
    }

    void App::onMotionDown(float x_pos, float y_pos)
    {
        DEBUG_INFO("Motion Down Event: [%.3f, %.3f]", x_pos, y_pos);
    }

    void App::onMotionUp(float x_pos, float y_pos)
    {
        DEBUG_INFO("Motion Up Event: [%.3f, %.3f]", x_pos, y_pos);
    }

    void App::onMotionMove(float x_pos, float y_pos)
    {
        DEBUG_INFO("Motion Move Event: [%.3f, %.3f]", x_pos, y_pos);
    }

    void App::onMotionCancel(float x_pos, float y_pos)
    {
        DEBUG_INFO("Motion Cancel Event: [%.3f, %.3f]", x_pos, y_pos);
    }

}