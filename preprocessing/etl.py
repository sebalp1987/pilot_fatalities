from resources.spark import SparkJob
from resources import STRING

class preProcessing(SparkJob):

    def __init__(self, train_file=True):
        self._spark = self.get_spark_session("etl")
        self._train_file = train_file

    def run(self):
        df = self.extract()
        df = self.transform(df)
        self.load(df)
        self._spark.stop()

    def extract(self):
        if self._train_file:
            df = (self._spark.read.csv(STRING.train, header=True, sep=','))
        else:
            df = (self._spark.read.csv(STRING.test, header=True, sep=','))

        return df

    @staticmethod
    def transform(df):
        df.show()
        return df

    def load(self, df):
        if self._train_file:
            df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(
                STRING.train_processed)
        else:
            df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(
                STRING.test_processed)


if __name__ == '__main__':
    preProcessing(train_file=True).run()


