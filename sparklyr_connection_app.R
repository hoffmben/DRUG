library(shiny)
library(ggplot2)
library(dplyr)
library(sparklyr)

ui <- fluidPage(
  
   titlePanel("Sparklyr Example"),
   
    sidebarLayout(
      sidebarPanel(
        selectInput("source", 
                    "Sparklyr Source", 
                    choices = c("MySQL", "S3"),
                    selected = "mysql")
      ),
      
      mainPanel(
        
        tabsetPanel(
          tabPanel("Plot", plotOutput("spark_plot")),
          tabPanel("Predictions", tableOutput("spark_predict"))
        )
      )
   )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  options(spark.install.dir = getwd())
  
  Sys.setenv(R_CONFIG_ACTIVE = "s3")
  s3config <- config::get()
  
  Sys.setenv("AWS_ACCESS_KEY_ID" = s3config$key,
             "AWS_SECRET_ACCESS_KEY" = s3config$secret,
             "AWS_DEFAULT_REGION" = "us-east-2")
  
  Sys.setenv(R_CONFIG_ACTIVE = "default")
  mysql_config <- config::get()

  sc <- spark_connect(master = "local")
  
  
  kmeans_model <- reactiveVal()
  
  iris_tbl <- reactiveVal()
  
  observeEvent(input$source, {
    if (input$source == "MySQL") {
      tryCatch(
        iris_tbl(spark_read_jdbc(
          sc,
          "mysql_iris",
          options = list(
            url = paste0(
              "jdbc:mysql://",
              mysql_config$host,
              ":",
              mysql_config$port,
              "/demo"
            ),
            user = mysql_config$username,
            password = mysql_config$password,
            dbtable = "iris",
            memory = TRUE
          )
        )),
        error = function(e){
          iris_tbl(spark_read_jdbc(
            sc,
            "mysql_iris",
            options = list(
              url = paste0(
                "jdbc:mysql://",
                mysql_config$host,
                ":",
                mysql_config$port,
                "/demo"
              ),
              user = mysql_config$username,
              password = mysql_config$password,
              dbtable = "iris",
              memory = TRUE
            )
          ))
        })
      
    } else {
      #Get spark context
      ctx <- spark_context(sc)
      
      #Use below to set the java spark context
      jsc <- invoke_static(sc,
                           "org.apache.spark.api.java.JavaSparkContext",
                           "fromSparkContext",
                           ctx)
      #set the s3 configs:
      hconf <- jsc %>% invoke("hadoopConfiguration")
      hconf %>% invoke("set", "fs.s3a.access.key", s3config$key)
      hconf %>% invoke("set", "fs.s3a.secret.key", s3config$secret)
      hconf %>% invoke("set", "fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
      
      iris_tbl(spark_read_csv(sc, "iris_csv",
                                 path = "s3a://drugdemo/iris.csv",
                                 memory = TRUE))
    }
    
    kmeans_model(iris_tbl() %>%
      ml_kmeans(k = 3, features = c("Petal_Width", "Petal_Length")))
  })
  
   output$spark_plot <- renderPlot({
     ml_predict(kmeans_model()) %>%
       collect() %>%
       ggplot(aes(Petal_Length, Petal_Width)) +
       geom_point(aes(Petal_Width, Petal_Length, col = factor(prediction + 1)),
                  size = 2, alpha = 0.5) +
       geom_point(data = kmeans_model()$centers, aes(Petal_Width, Petal_Length),
                  col = scales::muted(c("red", "green", "blue")),
                  pch = 'x', size = 12) +
       scale_color_discrete(name = "Predicted Cluster",
                            labels = paste("Cluster", 1:3)) +
       labs(
         x = "Petal Length",
         y = "Petal Width",
         title = "K-Means Clustering",
         subtitle = paste0("Spark Connection: ", input$source)
       )
   })


   output$spark_predict <- renderTable({
     predicted <- ml_predict(kmeans_model(), iris_tbl()) %>%
       collect
     as.data.frame.matrix(table(predicted$Species, predicted$prediction))
   }, rownames = TRUE)
}

# Run the application 
shinyApp(ui = ui, server = server)

