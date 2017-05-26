# carControl
Đây là mã nguồn chạy xe trên jetson TK1 đã được chuyển giao cho các đội thi.

# Cấu trúc mã nguồn
- lane_detection/: Thư viện chứa các thuật toán xác định đường, điểm mục tiêu (điểm vanishing point và lane center), thuật toán tracking, code test thuật toán vanishing point.
- main_pid/: Một số chương trình ví dụ để chạy xe.
- peripheral_driver/: Mã nguồn phần điều khiển của i2c và uart.
- stereo_vision/: Mã nguồn phần thu dữ liệu từ camera (hiện hỗ trợ thu dữ liệu từ camera Xbox 360 và Orbbec Astra dùng openNI2). Các thuật toán xử lý ảnh depth và xác định vật cản.

# Lưu ý quan trọng:
1. Mã nguồn hiện được cài đặt dành riêng cho các xe được gửi tới cho các đội thi. Nếu bạn muốn sử dụng mã nguồn cho các thiết bị khác, bạn cần cài các driver và sửa lại đường dẫn của chúng trong các file cài đặt (CMakeLists.txt) trong dự án này.
2. Các hệ số điều khiển động cơ của từng động cơ là khác nhau (kể cả động cơ của 2 xe được phát cho các đội cũng có đôi chút khác biệt). Bạn phải sửa lại hệ số cho phù hợp với từng động cơ để đảm bảo xe hoạt động đúng theo ý muốn.
3. Jetson TK1 không có nguồn giữ đồng hồ. Để tránh xảy ra lỗi, bạn nên đặt lại thời gian chuẩn mỗi khi thời gian bị sai.
4. Động cơ có thể không còn hoạt động chính xác khi phải chạy ở giới hạn biên (vận tốc lớn nhất, nhỏ nhất, ...) Bạn nên để xe chạy trong 90% giới hạn của xe (tham số vận tốc nên để từ 25-50, quay sang trái tối đa 20 độ, quay sang phải tối đa 20 độ).

# Biên dịch và chạy mã nguồn
Dịch cả project bằng lệnh:
```sh
$ cd DriverlessCar/carControl/src/0.3
$ cmake .
$ make
```
Sau khi dịch, cá file binary được tạo ra ở thư mục `DriverlessCar/carControl/src/0.3/bin/Release` bao gồm:
- `test-autocar` chương trình chạy xe tự động sử dụng thuật toán tìm vanishing point (được dịch ra từ file main_pid/autocar.cpp và các file thư viện đi kèm), có các lựa chọn log lại video đã thu, log lại tham số pwm truyền xuống để sửa góc bánh lái. Chạy bằng cách gọi `sudo directoryTo/test-autocar x` trong đó x là tham số vận tốc. Ví dụ chạy `sudo ./test-autocar 32` ở thư mục `bin/Release` tức là chạy xe với vận tốc có tham số bằng 32, hoặc chạy ở bin bằng cách gọi `sudo Release/test-autocar 32`. Nếu bạn chọn ghi log, các file log sẽ được tạo ra ở cùng thư mục gọi lệch thực thi code (chạy ở `bin` sẽ ghi log ra `bin`, chạy ở `bin/Release` sẽ ghi log ra `bin/Release`).
- `run-straight` chương trình chạy xe với yêu cầu chạy bánh lái ở góc 0 độ so với vector chỉ phương của xe (được dịch ra từ file main_pid/runStraight.cpp và các file thư viện đi kèm). Có thể chạy `sudo directoryTo/run-straight x` với tham số vận tốc x hoặc bỏ trống x, xe sẽ chạy với vận tốc mặc định là 28. Nếu chạy file này mà xe không đi thẳng một cách tương đối, bạn cần sửa lại các tham số điều khiển góc quay của bánh xe (xem phần dưới) sao cho xe chạy chấp nhận được.
- `check-uart` chương trình hỗ trợ truyền tham số xuống uart và lấy phản hồi từ uart (được dịch ra từ file main_pid/checkUart.cpp và các file thư viện đi kèm). Chạy không cần truyền tham số.
- `test-steering-coef` chương trình hỗ trợ chạy xe với các tham số điều khiển góc quay của bánh xe có thể thay đổi được dịch ra từ file main_pid/test-steering-coef.cpp và các file thư viện đi kèm). Chạy không cần truyền tham số. Khi bạn gọi chương trình này, dùng các phím tương ứng để thay đổi tham số của xe. Giảm tham số `pwm2` đến khi bánh xe không quay nữa thì đó là STEERING_MAX_RIGHT, bánh xe ở giữa thì đó là STEERING_NEUTRAL còn tăng `pwm2` bánh xe không quay nữa đó là STEERING_MAX_LEFT. Ba tham số này được define trong `peripheral_driver/i2c/api_i2c_pwm`. Theo kinh nghiệm, giới hạn của xe chỉ là ước lượng, bạn nên chỉnh các hệ số sao cho xe vẫn chạy được, `run-straight` thì xe chạy thẳng và STEERING_NEUTRAL bằng trung bình cộng của STEERING_MAX_LEFT và STEERING_MAX_RIGHT.
