mod handlers;
mod routes;

use handlers::app_state;

#[tokio::main]
async fn main() {
    let state = app_state();
    let app = routes::router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("绑定端口 3000 失败");

    println!("服务已启动: http://{}", listener.local_addr().unwrap());

    axum::serve(listener, app)
        .await
        .expect("服务启动失败");
}
